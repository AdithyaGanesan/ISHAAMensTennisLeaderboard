"""
Computes singles and doubles leaderboards from REAL scraped match data
(boys_matches.json, produced by usta_scraper.py).
"""

import json
import os
import datetime
import numpy as np
import requests

MATCHES_FILE = "boys_matches.json"

# Lovable Cloud endpoint config -- read from environment variables, never
# hardcoded here. Set these locally before running, and as GitHub Actions
# secrets for the scheduled version:
#   LEADERBOARD_ENDPOINT_URL = https://.../api/public/update-leaderboard
#   LEADERBOARD_UPDATE_SECRET = <the secret configured in Lovable Cloud>
LEADERBOARD_ENDPOINT_URL = os.environ.get("LEADERBOARD_ENDPOINT_URL")
LEADERBOARD_UPDATE_SECRET = os.environ.get("LEADERBOARD_UPDATE_SECRET")

K = 1.0          # upset-bonus weight -- starting point for real data
RIDGE = 1.0
ITERATIONS = 4
OUTCOME_WEIGHT = 6.0   # games-equivalent bonus/penalty for actually winning/losing
                        # the match (not just the game margin) -- roughly "one set"
                        # worth of weight, tunable. This exists specifically to
                        # stop a losing side from out-rating the side that beat
                        # them (the Powell/McGary case).
CONFIDENCE_PRIOR = 5.0  # "games played" damping constant. A player's displayed
                          # rating gets scaled by games/(games+CONFIDENCE_PRIOR),
                          # so someone with few matches is pulled toward zero
                          # instead of carrying full weight off a small sample.

# Position multiplier: a coach's lineup order is itself a skill signal --
# playing (and winning) at 1 singles should count for more than the same
# result at 3 singles. Applied to the combined margin+outcome score before
# the regression solve. All tunable, and intentionally NOT equally spaced
# (the drop from 1 to 2 is bigger than 2 to 3, reflecting that "1" is
# usually a team's clearly-best player while 2/3 are closer together).
SINGLES_POSITION_WEIGHTS = {1: 1.5, 2: 1.2, 3: 1.0}
DOUBLES_POSITION_WEIGHTS = {1: 1.25, 2: 1.0}


def games_from_sets(sets):
    """
    Turns a side's per-set score list into a total game count.
    Match tiebreaks (tiebreakSet=True) count as 1 game to whoever won it,
    per the agreed rule -- not their raw 10-point score.

    Defensive: a regular set can have setScore=null when it was never
    actually played (e.g. a retirement partway through the match). That
    set contributes 0 games rather than crashing -- it's not fabricating
    a score, just correctly recording "no games happened here."
    """
    total = 0
    for s in sets:
        if s["tiebreakSet"]:
            total += 1 if s["didWin"] else 0
        elif s["setScore"] is not None:
            total += int(s["setScore"])
        # else: unplayed set, contributes 0 -- no crash, no fabrication
    return total


def pair_id(names):
    return " & ".join(sorted(names))


def load_rows(matches_file):
    with open(matches_file) as f:
        matches = json.load(f)

    singles_rows = []
    doubles_rows = []
    team_lookup = {}   # abbreviation -> school name, for reporting

    for match in matches:
        for t in match["teams"]:
            team_lookup[t["abbreviation"]] = t.get("parentOrganisation", {}).get("name", t["abbreviation"])

        for slot in match["tieMatchUps"]:
            if slot["status"] != "COMPLETED":
                continue  # incomplete/missing slot -- skipped for now, no fallback yet

            side1, side2 = slot["side1"], slot["side2"]
            if not side1.get("score") or not side2.get("score"):
                continue

            games_a = games_from_sets(side1["score"]["sets"])
            games_b = games_from_sets(side2["score"]["sets"])
            margin_a = games_a - games_b
            margin_b = -margin_a

            if slot["type"] == "SINGLES":
                p_a = f"{side1['participants'][0]['firstName']} {side1['participants'][0]['lastName']}"
                p_b = f"{side2['participants'][0]['firstName']} {side2['participants'][0]['lastName']}"
                singles_rows.append({"entity": p_a, "margin": margin_a, "opponent": p_b,
                                      "team": side1["team"]["abbreviation"], "did_win": side1["didWin"],
                                      "position": slot["collectionPosition"]})
                singles_rows.append({"entity": p_b, "margin": margin_b, "opponent": p_a,
                                      "team": side2["team"]["abbreviation"], "did_win": side2["didWin"],
                                      "position": slot["collectionPosition"]})
            else:  # DOUBLES -- pair treated as one individual
                names_a = [f"{p['firstName']} {p['lastName']}" for p in side1["participants"]]
                names_b = [f"{p['firstName']} {p['lastName']}" for p in side2["participants"]]
                pid_a, pid_b = pair_id(names_a), pair_id(names_b)
                doubles_rows.append({"entity": pid_a, "margin": margin_a, "opponent": pid_b,
                                      "team": side1["team"]["abbreviation"], "did_win": side1["didWin"],
                                      "position": slot["collectionPosition"]})
                doubles_rows.append({"entity": pid_b, "margin": margin_b, "opponent": pid_a,
                                      "team": side2["team"]["abbreviation"], "did_win": side2["didWin"],
                                      "position": slot["collectionPosition"]})

    return singles_rows, doubles_rows, team_lookup


def solve_opr(rows, ridge=1.0):
    """rows need an 'entity' and a 'score' field (already outcome-weighted
    where applicable -- this function itself is agnostic to what 'score'
    means, it just does the ridge least-squares solve)."""
    names = sorted({r["entity"] for r in rows})
    idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(rows), len(names)))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        A[i, idx[r["entity"]]] = 1
        y[i] = r["score"]
    AtA = A.T @ A + ridge * np.eye(len(names))
    x = np.linalg.solve(AtA, A.T @ y)
    return dict(zip(names, x))


def games_played_counts(rows):
    counts = {}
    for r in rows:
        counts[r["entity"]] = counts.get(r["entity"], 0) + 1
    return counts


def rate_with_upset_bonus(rows, k=1.0, ridge=1.0, iterations=4,
                           outcome_weight=6.0, confidence_prior=5.0,
                           position_weights=None):
    # Build the base "score" used in the regression: raw game margin, PLUS
    # a fixed bonus/penalty for actually winning or losing the match, THEN
    # scaled by a position multiplier (winning at 1 singles counts for
    # more than winning at 3 singles). This stops a side that racked up
    # more games but still lost (see the Powell/McGary case) from
    # out-rating the side that beat them, and makes lineup position
    # matter the way a coach's own ordering implies it should.
    position_weights = position_weights or {}
    base_rows = []
    for r in rows:
        outcome_term = outcome_weight if r["did_win"] else -outcome_weight
        pos_weight = position_weights.get(r["position"], 1.0)
        base_rows.append({"entity": r["entity"], "score": (r["margin"] + outcome_term) * pos_weight})

    rating = solve_opr(base_rows, ridge=ridge)  # bootstrap pass

    for _ in range(iterations):
        adjusted = []
        for r, base in zip(rows, base_rows):
            my_rating = rating.get(r["entity"], 0.0)
            opp_rating = rating.get(r["opponent"], 0.0)
            gap = opp_rating - my_rating   # positive = tougher opponent
            bonus = k * gap
            adjusted.append({"entity": r["entity"], "score": base["score"] + bonus})
        rating = solve_opr(adjusted, ridge=ridge)

    # Non-additive step: scale each final rating by how many matches that
    # specific entity has actually played, so a small sample gets pulled
    # toward zero instead of carrying full weight.
    counts = games_played_counts(rows)
    confidence_scaled = {}
    for entity, raw_rating in rating.items():
        n = counts.get(entity, 0)
        confidence = n / (n + confidence_prior)
        confidence_scaled[entity] = raw_rating * confidence

    return confidence_scaled, counts


def entity_team(rows, entity_name):
    for r in rows:
        if r["entity"] == entity_name:
            return r["team"]
    return "?"


def top_n(d, n=15):
    return sorted(d.items(), key=lambda kv: -kv[1])[:n]


def build_payload(rating_dict, counts, rows):
    """Turns a {entity: rating} dict into the row shape the Lovable
    endpoint expects: entity_name, team, rating, matches_played."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    out = []
    for name, rating in rating_dict.items():
        abbr = entity_team(rows, name)
        out.append({
            "entity_name": name,
            "team": abbr,
            "rating": round(float(rating), 3),
            "matches_played": counts.get(name, 0),
            "last_updated": now_iso,
        })
    return out


def push_to_lovable(singles_payload, doubles_payload):
    """
    POSTs the computed leaderboards to the Lovable Cloud endpoint. Skips
    cleanly (with a clear message) if the URL/secret env vars aren't set --
    this lets the script still be run locally just to look at the printed
    leaderboard without needing the website side configured yet.
    """
    if not LEADERBOARD_ENDPOINT_URL or not LEADERBOARD_UPDATE_SECRET:
        print("\n[skip] LEADERBOARD_ENDPOINT_URL / LEADERBOARD_UPDATE_SECRET "
              "not set in the environment -- not pushing to the website. "
              "(This is fine if you're just checking the leaderboard locally.)")
        return

    body = {
        "secret": LEADERBOARD_UPDATE_SECRET,
        "singles": singles_payload,
        "doubles": doubles_payload,
    }

    print(f"\nPushing {len(singles_payload)} singles + "
          f"{len(doubles_payload)} doubles rows to {LEADERBOARD_ENDPOINT_URL} ...")
    try:
        resp = requests.post(LEADERBOARD_ENDPOINT_URL, json=body, timeout=30)
    except Exception as e:
        print(f"[FAILED] Could not reach the endpoint: {e}")
        return

    if resp.status_code == 401:
        print("[FAILED] 401 Unauthorized -- the secret sent doesn't match "
              "what's configured in Lovable. Double-check "
              "LEADERBOARD_UPDATE_SECRET.")
        return
    if not resp.ok:
        print(f"[FAILED] HTTP {resp.status_code}: {resp.text[:500]}")
        return

    print(f"[OK] Website updated successfully. Response: {resp.text[:200]}")


def main():
    singles_rows, doubles_rows, team_lookup = load_rows(MATCHES_FILE)

    print(f"Loaded {len(singles_rows)//2} singles matches, "
          f"{len(doubles_rows)//2} doubles matches.\n")

    if not singles_rows and not doubles_rows:
        print("No completed slots found -- nothing to rate.")
        return

    singles_payload, doubles_payload = [], []

    if singles_rows:
        singles_rating, singles_counts = rate_with_upset_bonus(
            singles_rows, k=K, ridge=RIDGE, iterations=ITERATIONS,
            outcome_weight=OUTCOME_WEIGHT, confidence_prior=CONFIDENCE_PRIOR,
            position_weights=SINGLES_POSITION_WEIGHTS)
        print("=" * 60)
        print(f"SINGLES LEADERBOARD (k={K}, ridge={RIDGE}, "
              f"outcome_weight={OUTCOME_WEIGHT}, confidence_prior={CONFIDENCE_PRIOR})")
        print("=" * 60)
        for name, rating in top_n(singles_rating):
            abbr = entity_team(singles_rows, name)
            school = team_lookup.get(abbr, abbr)
            n = singles_counts[name]
            print(f"{rating:+6.2f}  {name:<20s} ({school}, {n} match{'es' if n != 1 else ''})")
        singles_payload = build_payload(singles_rating, singles_counts, singles_rows)

    if doubles_rows:
        doubles_rating, doubles_counts = rate_with_upset_bonus(
            doubles_rows, k=K, ridge=RIDGE, iterations=ITERATIONS,
            outcome_weight=OUTCOME_WEIGHT, confidence_prior=CONFIDENCE_PRIOR,
            position_weights=DOUBLES_POSITION_WEIGHTS)
        print()
        print("=" * 60)
        print(f"DOUBLES LEADERBOARD (pairs as individuals, k={K}, ridge={RIDGE}, "
              f"outcome_weight={OUTCOME_WEIGHT}, confidence_prior={CONFIDENCE_PRIOR})")
        print("=" * 60)
        for name, rating in top_n(doubles_rating):
            abbr = entity_team(doubles_rows, name)
            school = team_lookup.get(abbr, abbr)
            n = doubles_counts[name]
            print(f"{rating:+6.2f}  {name:<28s} ({school}, {n} match{'es' if n != 1 else ''})")
        doubles_payload = build_payload(doubles_rating, doubles_counts, doubles_rows)

    push_to_lovable(singles_payload, doubles_payload)


if __name__ == "__main__":
    main()