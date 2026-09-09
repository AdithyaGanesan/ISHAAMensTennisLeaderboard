#This file tests scraping capabilities

import json
import os
import time
import datetime
import requests

GQL_URL = "https://prd-usta-kube.clubspark.pro/mesh-api/graphql"
SEEN_IDS_FILE = "seen_match_ids.json"
MATCHES_FILE = "boys_matches.json"

PAGE_SIZE = 50            # items per getStateResults call
REQUEST_DELAY_SECONDS = 0.4   # be polite -- small pause between calls

# Indiana's org ID within Serve Tennis, captured from the live request.
INDIANA_ORG_ID = "00000001-0006-0043-0000-000000000000"

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
      ...StateResultItem_QueryFragment
    }
    totalItems
  }
}

fragment StateResultItem_QueryFragment on desk_DualMatch {
  id
  gender
  providerOrganisations {
    name
  }
  startDateTime {
    dateTimeString
    noScheduledTime
    timezoneName
  }
  homeTeam {
    id
  }
  teams {
    id
    name
    displayName
    abbreviation
    score
    logo {
      url
    }
    displayOrder
    division
    didWin
    sideNumber
    parentOrganisation {
      id
      name
      parentOrganisation {
        id
        name
        urlSlug
      }
      organisation {
        profileImage {
          medium
        }
      }
    }
  }
  scoringFormat
  tournamentDeskUrl
  webLinks {
    url
    name
  }
  isConferenceMatch
}
"""

DUAL_MATCH_QUERY = """
query dualMatch($id: ID!) {
  dualMatch(id: $id) {
    id
    startDateTime {
      timezoneName
      dateTimeString
      noScheduledTime
    }
    homeTeam {
      id
    }
    teams {
      id
      name
      score
      didWin
      sideNumber
      abbreviation
      division
      conference
      displayOrder
      parentOrganisation {
        name
        organisation {
          profileImage {
            small
          }
        }
      }
    }
    scoringFormat
    tieFormat {
      name
      collectionDefinitions {
        name
        id
        matchUpType
      }
    }
    tieMatchUps {
      type
      status
      collectionId
      collectionPosition
      side1 {
        team {
          abbreviation
        }
        displayStatus {
          status
          subStatus
        }
        participants {
          personId
          firstName
          lastName
        }
        score {
          scoreString
          sets {
            setScore
            tiebreakSet
            tiebreakScore
            didWin
          }
        }
        didWin
      }
      side2 {
        team {
          abbreviation
        }
        displayStatus {
          status
          subStatus
        }
        participants {
          personId
          firstName
          lastName
        }
        score {
          scoreString
          sets {
            setScore
            tiebreakSet
            tiebreakScore
            didWin
          }
        }
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
        raise RuntimeError(f"GraphQL error: {payload['errors']}")
    return payload["data"]


def load_seen_ids():
    if os.path.exists(SEEN_IDS_FILE):
        with open(SEEN_IDS_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(ids):
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(sorted(ids), f, indent=2)


def load_matches():
    if os.path.exists(MATCHES_FILE):
        with open(MATCHES_FILE) as f:
            return json.load(f)
    return []


def save_matches(matches):
    with open(MATCHES_FILE, "w") as f:
        json.dump(matches, f, indent=2)


def fetch_new_match_summaries(seen_ids):
    """
    Pages through getStateResults newest-first. Stops once an entire page
    comes back with zero unseen match IDs (a full page of "already have
    this" is a safe stopping point even with same-timestamp ties, unlike
    stopping at the very first seen ID).
    """
    new_summaries = []
    skip = 0
    now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")

    while True:
        variables = {
            "skip": skip,
            "limit": PAGE_SIZE,
            "filters": {
                "isCompleted": True,
                "startDate": {"lt": now_iso},
                "teamOrgParentIds": [INDIANA_ORG_ID],
            },
            "sortOrder": "DESCENDING",
        }
        data = gql(STATE_RESULTS_QUERY, variables, "getStateResults")
        page = data["dualMatchesPaginated"]["items"]
        total = data["dualMatchesPaginated"]["totalItems"]

        if not page:
            break

        boys_page = [m for m in page if m.get("gender") == "MALE"]
        page_new = [m for m in boys_page if m["id"] not in seen_ids]

        print(f"skip={skip}: page had {len(page)} items "
              f"({len(boys_page)} boys), {len(page_new)} new "
              f"(total available: {total})")

        new_summaries.extend(page_new)

        if not page_new:
            # Nothing new on this whole page -- safe to stop.
            break

        skip += PAGE_SIZE
        time.sleep(REQUEST_DELAY_SECONDS)

    return new_summaries


def fetch_full_match(match_id):
    data = gql(DUAL_MATCH_QUERY, {"id": match_id}, "dualMatch")
    return data["dualMatch"]


def main():
    seen_ids = load_seen_ids()
    matches = load_matches()

    print(f"Currently have {len(seen_ids)} matches captured. Checking for new ones...\n")
    new_summaries = fetch_new_match_summaries(seen_ids)

    if not new_summaries:
        print("\nNo new completed boys matches found. Nothing to do.")
        return

    print(f"\nFound {len(new_summaries)} new match(es). Fetching full detail for each...")

    for i, summary in enumerate(new_summaries):
        match_id = summary["id"]
        print(f"  [{i+1}/{len(new_summaries)}] {match_id}")
        try:
            full = fetch_full_match(match_id)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        full["_gender"] = summary.get("gender")  # carry over from list query
        matches.append(full)
        seen_ids.add(match_id)
        time.sleep(REQUEST_DELAY_SECONDS)

    save_matches(matches)
    save_seen_ids(seen_ids)
    print(f"\nDone. {MATCHES_FILE} now has {len(matches)} total matches.")


if __name__ == "__main__":
    main()