
import json
import requests

GQL_URL = "https://prd-usta-kube.clubspark.pro/mesh-api/graphql"
INDIANA_ORG_ID = "00000001-0006-0043-0000-000000000000"

TEAM_A_SEARCH = "columbus north"
TEAM_B_SEARCH = "seymour"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

STATE_RESULTS_QUERY = """
query getStateResults($skip: Int!, $limit: Int!, $filters: desk_DualMatchesFilter, $sortOrder: desk_SortDirection!) {
  dualMatchesPaginated(
    skip: $skip
    limit: $limit
    filter: $filters
    sort: {field: START_DATE, direction: $sortOrder}
  ) {
    items {
      id
      gender
      startDateTime { dateTimeString }
      teams { displayName name score didWin }
    }
    totalItems
  }
}
"""

DUAL_MATCH_QUERY = """
query dualMatch($id: ID!) {
  dualMatch(id: $id) {
    id
    startDateTime { dateTimeString timezoneName }
    teams {
      name
      score
      didWin
      abbreviation
      parentOrganisation { name }
    }
    tieMatchUps {
      type
      status
      collectionPosition
      side1 {
        team { abbreviation }
        participants { firstName lastName }
        score { scoreString sets { setScore tiebreakSet tiebreakScore didWin } }
        didWin
      }
      side2 {
        team { abbreviation }
        participants { firstName lastName }
        score { scoreString sets { setScore tiebreakSet tiebreakScore didWin } }
        didWin
      }
    }
  }
}
"""


def gql(query, variables, operation_name):
    resp = requests.post(
        GQL_URL,
        headers=HEADERS,
        json={"query": query, "variables": variables, "operationName": operation_name},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        print("GraphQL errors:", json.dumps(payload["errors"], indent=2))
        raise RuntimeError("GraphQL error -- see above")
    return payload["data"]


def find_match(team_a, team_b, max_pages=10, page_size=50):
    """Pages through completed matches (newest first) looking for both
    team names appearing in the same match."""
    skip = 0
    for page_num in range(max_pages):
        variables = {
            "skip": skip,
            "limit": page_size,
            "filters": {
                "isCompleted": True,
                "startDate": {"lt": "2026-08-20T00:00:00.000Z"},
                "teamOrgParentIds": [INDIANA_ORG_ID],
            },
            "sortOrder": "DESCENDING",
        }
        data = gql(STATE_RESULTS_QUERY, variables, "getStateResults")
        items = data["dualMatchesPaginated"]["items"]
        if not items:
            break

        for m in items:
            names = " ".join((t.get("displayName") or t.get("name") or "").lower()
                              for t in m["teams"])
            if team_a in names and team_b in names:
                return m

        skip += page_size
        print(f"  ...checked {skip} matches so far, not found yet")

    return None


def main():
    print(f"Searching for a completed match between "
          f"'{TEAM_A_SEARCH}' and '{TEAM_B_SEARCH}'...\n")

    summary = find_match(TEAM_A_SEARCH, TEAM_B_SEARCH)

    if not summary:
        print("\nNo match found containing both team names. Possible reasons:\n"
              "  - It hasn't been marked completed / entered yet\n"
              "  - Team name spelling on the site differs from what's "
              "searched for here (check the printed near-misses, if any)\n"
              "  - It's further back than max_pages*page_size results checked")
        return

    print(f"\nFound it: {[t.get('displayName') or t.get('name') for t in summary['teams']]}")
    print(f"Date: {summary['startDateTime']['dateTimeString']}")
    print(f"Match id: {summary['id']}\n")

    print("=" * 60)
    print("FULL MATCH DATA")
    print("=" * 60)

    full = gql(DUAL_MATCH_QUERY, {"id": summary["id"]}, "dualMatch")["dualMatch"]
    print(json.dumps(full, indent=2))

    print("\n" + "=" * 60)
    print("SLOT COMPLETENESS CHECK")
    print("=" * 60)
    expected_slots = [("SINGLES", 1), ("SINGLES", 2), ("SINGLES", 3),
                       ("DOUBLES", 1), ("DOUBLES", 2)]
    found_slots = {(m["type"], m["collectionPosition"]) for m in full["tieMatchUps"]}
    for slot_type, pos in expected_slots:
        mark = "present" if (slot_type, pos) in found_slots else "MISSING"
        print(f"  {slot_type} #{pos}: {mark}")


if __name__ == "__main__":
    main()