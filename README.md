# ISHAA Men's Tennis Leaderboard

A form-based singles and doubles leaderboard for Indiana high school boys
varsity tennis, built on real match results reported through USTA Serve
Tennis. Rankings are inspired by OPR (Offensive Power Rating), the method
FIRST Robotics Competition uses to isolate individual contribution from
shared results — adapted here for how high school tennis actually works:
singles, doubles, lineup position, and season-long form rather than a fixed
skill rating.

**Live site:** https://intennisleaderboard.lovable.app/

## How it works

1. **`tTest.py`** pulls completed match results directly from USTA
   Serve Tennis's public API (no browser required — it replays the same
   GraphQL calls the site itself uses). It's incremental: each run only
   fetches matches it hasn't already seen, tracked in `seen_match_ids.json`.
2. **`tLeaderboard.py`** reads the scraped data
   (`boys_matches.json`) and computes a rating for every singles player and
   every doubles pair (pairs are rated as a single unit, not split into
   individual credit). It then pushes the results to the live site via an
   API endpoint.
3. **`BradleyTerry.py`** reads the scraped data and computes a rating for every
   singles player and doubles pair. An attempt to improve the flawed rating system
   created by tLeaderboard.py, specifically addressing gaps in match toughness. 
   Uses a Bradley-Terry model to estimate latent individual strength from data.

4. **GitHub Actions** (`.github/workflows/update-leaderboard.yml`) runs both
   scripts automatically, twice a day, and commits the updated data files
   back to this repo so state persists between runs.
5. **Lovable** hosts the frontend, reading the rating data pushed to it by
   step 2.

See the website for the methodology for calculations. 
## Repo structure

```
TennisLeaderboard/
  tTest.py                      scrapes new match results from USTA
  tLeaderboard.py               computes ratings, pushes to the website
  boys_matches.json             accumulated raw match data (auto-updated)
  seen_match_ids.json           tracks which matches have been scraped
.github/workflows/
  update-leaderboard.yml        runs the above on a schedule
```

## Running it locally

```
cd TennisLeaderboard
pip install requests numpy
python3 usta_scraper.py
python3 compute_leaderboards_real.py
```

To actually push results to the live site, set these environment variables
first. 

```
LEADERBOARD_ENDPOINT_URL=<lovable endpoint>
LEADERBOARD_UPDATE_SECRET=<shared secret>
```

Without them set, the script still runs and prints the leaderboard locally. 

## Data source

Match data comes from USTA Serve Tennis's public results pages
(`highschools.usta.com/state/indiana`). 

This project is not affiliated with or endorsed by USTA.

## About

Built by [Adithya Ganesan](https://github.com/AdithyaGanesan)
