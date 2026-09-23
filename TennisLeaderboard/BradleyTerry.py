import json
import os
import datetime
import numpy as np
from scipy.optimize import minimize
import requests
 
MATCHES_FILE = "boys_matches.json"
 
LEADERBOARD_ENDPOINT_URL = os.environ.get("LEADERBOARD_ENDPOINT_URL")
LEADERBOARD_UPDATE_SECRET = os.environ.get("LEADERBOARD_UPDATE_SECRET")
 
OUTCOME_PSEUDO_GAMES = 4.0   # "virtual" games added to the actual match
                              # winner's tally before fitting -- makes sure
                              # winning the match always matters beyond raw
                              # game count, same purpose as the old
                              # OUTCOME_WEIGHT but expressed as pseudo-data
                              # instead of an additive score term.
 
SINGLES_POSITION_WEIGHTS = {1: 2.0, 2: 1.5, 3: 1.0}
DOUBLES_POSITION_WEIGHTS = {1: 1.5, 2: 1.0}
 
L2_REG = 0.02   # small amount kept purely for numerical identifiability
                 # (Bradley-Terry ratings are only defined up to a shared
                 # additive constant otherwise). The main small-sample
                 # protection is now PRIOR_GAMES below, not this.
 
PRIOR_GAMES = 100.0   # Every entity is treated as if they've already played
                        # this many games at a dead-even 50/50 against a
                        # perfectly average opponent, before their real
                        # results are added in. A single real match --
                        # blowout or close -- carries roughly 20-30
                        # "effective games" of weight (after the outcome
                        # bonus and position multiplier), so 100 is sized
                        # to require somewhere around 4-5 real matches of
                        # evidence before this prior's influence fades
                        # substantially.
 
 
def games_from_sets(sets):
    """
    Turns a side's per-set score list into a total game count.
    Match tiebreaks (tiebreakSet=True) count as 1 game to whoever won it.
    A regular set with setScore=null (never played -- e.g. a retirement)
    contributes 0 rather than crashing.
    """
    total = 0
    for s in sets:
        if s["tiebreakSet"]:
            total += 1 if s["didWin"] else 0
        elif s["setScore"] is not None:
            total += int(s["setScore"])
    return total
 
 
def pair_id(names):
    return " & ".join(sorted(names))
 
 
def load_match_records(matches_file):
    """
    One record per completed slot (NOT one per side -- Bradley-Terry
    naturally handles both directions of a single A-vs-B observation, no
    need to duplicate rows the way the old linear model did).
    """
    with open(matches_file) as f:
        matches = json.load(f)
 
    singles_records = []
    doubles_records = []
    team_lookup = {}   # kept only as a loose fallback, not the source of truth
 
    for match in matches:
        # Resolve team names by STRUCTURAL POSITION (sideNumber), not by
        # abbreviation string matching. Two different schools can share an
        # abbreviation (e.g. two schools both landing on "CHS"), and if
        # they happen to play EACH OTHER, an abbreviation-keyed dict only
        # has room for one of them -- both sides would silently resolve to
        # the same (wrong, for one of them) school name. sideNumber is a
        # structural field (1 or 2), not a string that can collide, so
        # this is safe even in that worst case.
        side_number_to_name = {
            t["sideNumber"]: t.get("parentOrganisation", {}).get("name", t["abbreviation"])
            for t in match["teams"]
        }
        team_a_name = side_number_to_name.get(1, "?")
        team_b_name = side_number_to_name.get(2, "?")
        for t in match["teams"]:
            team_lookup[t["abbreviation"]] = t.get("parentOrganisation", {}).get("name", t["abbreviation"])
 
        for slot in match["tieMatchUps"]:
            if slot["status"] != "COMPLETED":
                continue
            side1, side2 = slot["side1"], slot["side2"]
            if not side1.get("score") or not side2.get("score"):
                continue
 
            games_a = games_from_sets(side1["score"]["sets"])
            games_b = games_from_sets(side2["score"]["sets"])
            if games_a == games_b:
                continue  # shouldn't happen in a completed match, but skip defensively
 
            record_base = {
                "games_a": games_a, "games_b": games_b,
                "did_win_a": side1["didWin"],
                "position": slot["collectionPosition"],
                # side1/side2 in tieMatchUps always correspond to
                # sideNumber 1/2 for the whole match -- structural, not
                # string-matched, so this is safe even if both teams
                # happen to share the same abbreviation.
                "team_a": team_a_name,
                "team_b": team_b_name,
            }
 
            if slot["type"] == "SINGLES":
                entity_a = f"{side1['participants'][0]['firstName']} {side1['participants'][0]['lastName']}"
                entity_b = f"{side2['participants'][0]['firstName']} {side2['participants'][0]['lastName']}"
                singles_records.append({**record_base, "entity_a": entity_a, "entity_b": entity_b})
            else:
                names_a = [f"{p['firstName']} {p['lastName']}" for p in side1["participants"]]
                names_b = [f"{p['firstName']} {p['lastName']}" for p in side2["participants"]]
                doubles_records.append({**record_base, "entity_a": pair_id(names_a), "entity_b": pair_id(names_b)})
 
    return singles_records, doubles_records, team_lookup
 
 
def fit_bradley_terry(records, position_weights, l2_reg=0.02,
                       outcome_pseudo_games=4.0, prior_games=6.0):
    """
    Fits one rating per entity by maximum likelihood.
 
    For each match: k = effective games won by entity_a (out of n effective
    trials), p = sigmoid(rating_a - rating_b) is the model's implied
    probability entity_a wins any given game. We maximize the binomial
    log-likelihood of the observed game counts.
 
    "Effective" games/trials = raw games, with OUTCOME_PSEUDO_GAMES added
    to the actual winner's side, then everything scaled by the position
    weight for that slot (scaling win-count and trial-count together keeps
    the implied win probability the same, it just makes that observation
    count for more in the overall fit).
 
    Small-sample protection: every entity additionally gets PRIOR_GAMES
    "games" at a fixed 50/50 against a locked rating of 0 (an "average"
    reference opponent, not a free parameter). A player's real results
    have to accumulate past this synthetic baseline before their rating
    can move far from zero -- one blowout with little else behind it stays
    mostly anchored.
    """
    if not records:
        return {}, {}
 
    entities = sorted({r["entity_a"] for r in records} | {r["entity_b"] for r in records})
    idx = {e: i for i, e in enumerate(entities)}
    n_entities = len(entities)
 
    idx_a = np.array([idx[r["entity_a"]] for r in records])
    idx_b = np.array([idx[r["entity_b"]] for r in records])
 
    k = np.zeros(len(records))   # effective wins for entity_a
    n = np.zeros(len(records))   # effective trials
    for i, r in enumerate(records):
        ga, gb = r["games_a"], r["games_b"]
        if r["did_win_a"]:
            ga += outcome_pseudo_games
        else:
            gb += outcome_pseudo_games
        w = position_weights.get(r["position"], 1.0)
        k[i] = ga * w
        n[i] = (ga + gb) * w
 
    def neg_log_likelihood_and_grad(params):
        d = params[idx_a] - params[idx_b]
        p = 1.0 / (1.0 + np.exp(-d))
        p = np.clip(p, 1e-9, 1 - 1e-9)  # avoid log(0)
 
        ll = np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
        ll -= l2_reg * np.sum(params ** 2)
 
        # Gradient: standard logistic/Bradley-Terry form -- observed wins
        # minus expected wins, scattered back to each entity's parameter.
        residual = k - n * p   # positive = entity_a overperformed vs. model
        grad = np.zeros(n_entities)
        np.add.at(grad, idx_a, residual)
        np.add.at(grad, idx_b, -residual)
 
        # Prior-games term: every entity vs. a FIXED rating of 0, at a
        # locked 50/50 split (k_prior = n_prior/2). d = params[i] - 0.
        p_prior = 1.0 / (1.0 + np.exp(-params))
        p_prior = np.clip(p_prior, 1e-9, 1 - 1e-9)
        k_prior = prior_games / 2.0
        ll += np.sum(k_prior * np.log(p_prior) + k_prior * np.log(1 - p_prior))
        grad += (k_prior - prior_games * p_prior)
 
        grad -= 2 * l2_reg * params
 
        return -ll, -grad  # minimize negative log-likelihood
 
    x0 = np.zeros(n_entities)
    result = minimize(neg_log_likelihood_and_grad, x0, jac=True, method="L-BFGS-B")
 
    ratings = dict(zip(entities, result.x))
 
    counts = {}
    for r in records:
        counts[r["entity_a"]] = counts.get(r["entity_a"], 0) + 1
        counts[r["entity_b"]] = counts.get(r["entity_b"], 0) + 1
 
    return ratings, counts
 
 
def entity_team(records, entity_name):
    for r in records:
        if r["entity_a"] == entity_name:
            return r["team_a"]
        if r["entity_b"] == entity_name:
            return r["team_b"]
    return "?"
 
 
def top_n(d, n=20):
    return sorted(d.items(), key=lambda kv: -kv[1])[:n]
 
 
def build_payload(rating_dict, counts, records):
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    out = []
    for name, rating in rating_dict.items():
        abbr = entity_team(records, name)
        out.append({
            "entity_name": name,
            "team": abbr,
            "rating": round(float(rating), 4),
            "matches_played": counts.get(name, 0),
            "last_updated": now_iso,
        })
    return out
 
 
def push_to_lovable(singles_payload, doubles_payload):
    if not LEADERBOARD_ENDPOINT_URL or not LEADERBOARD_UPDATE_SECRET:
        print("\n[skip] LEADERBOARD_ENDPOINT_URL / LEADERBOARD_UPDATE_SECRET "
              "not set -- not pushing to the website.")
        return
 
    body = {"secret": LEADERBOARD_UPDATE_SECRET, "singles": singles_payload, "doubles": doubles_payload}
    print(f"\nPushing {len(singles_payload)} singles + {len(doubles_payload)} "
          f"doubles rows to {LEADERBOARD_ENDPOINT_URL} ...")
    try:
        resp = requests.post(LEADERBOARD_ENDPOINT_URL, json=body, timeout=30)
    except Exception as e:
        print(f"[FAILED] Could not reach the endpoint: {e}")
        return
 
    if resp.status_code == 401:
        print("[FAILED] 401 Unauthorized -- secret mismatch.")
        return
    if not resp.ok:
        print(f"[FAILED] HTTP {resp.status_code}: {resp.text[:500]}")
        return
    print(f"[OK] Website updated successfully. Response: {resp.text[:200]}")
 
 
def main():
    singles_records, doubles_records, team_lookup = load_match_records(MATCHES_FILE)
    print(f"Loaded {len(singles_records)} singles matches, "
          f"{len(doubles_records)} doubles matches.\n")
 
    if not singles_records and not doubles_records:
        print("No completed slots found -- nothing to rate.")
        return
 
    singles_payload, doubles_payload = [], []
 
    if singles_records:
        singles_rating, singles_counts = fit_bradley_terry(
            singles_records, SINGLES_POSITION_WEIGHTS, l2_reg=L2_REG,
            outcome_pseudo_games=OUTCOME_PSEUDO_GAMES, prior_games=PRIOR_GAMES)
        print("=" * 60)
        print(f"SINGLES LEADERBOARD (Bradley-Terry MLE, l2_reg={L2_REG})")
        print("=" * 60)
        for name, rating in top_n(singles_rating):
            school = entity_team(singles_records, name)
            n = singles_counts[name]
            print(f"{rating:+7.3f}  {name:<20s} ({school}, {n} match{'es' if n != 1 else ''})")
        singles_payload = build_payload(singles_rating, singles_counts, singles_records)
 
    if doubles_records:
        doubles_rating, doubles_counts = fit_bradley_terry(
            doubles_records, DOUBLES_POSITION_WEIGHTS, l2_reg=L2_REG,
            outcome_pseudo_games=OUTCOME_PSEUDO_GAMES, prior_games=PRIOR_GAMES)
        print()
        print("=" * 60)
        print(f"DOUBLES LEADERBOARD (pairs as individuals, Bradley-Terry MLE, l2_reg={L2_REG})")
        print("=" * 60)
        for name, rating in top_n(doubles_rating):
            school = entity_team(doubles_records, name)
            n = doubles_counts[name]
            print(f"{rating:+7.3f}  {name:<28s} ({school}, {n} match{'es' if n != 1 else ''})")
        doubles_payload = build_payload(doubles_rating, doubles_counts, doubles_records)
 
    push_to_lovable(singles_payload, doubles_payload)
 
 
if __name__ == "__main__":
    main()
 