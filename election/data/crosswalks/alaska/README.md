# Alaska precinct to borough crosswalk

Alaska has no counties. Its county equivalents are 19 organized **boroughs** and
11 **census areas** (the Census Bureau's divisions of the Unorganized Borough),
30 in all. The Division of Elections reports results by precinct and House
district, never by borough, so these files assign precincts to boroughs and
census areas. Reports for Alaska are therefore built in this order: statewide,
each borough or census area, then House district and Senate district.

| File | One row per | Use |
| --- | --- | --- |
| `precinct-borough.csv` | precinct and borough it overlaps | Election-day precinct results → borough |
| `house-district-borough.csv` | House district and borough it overlaps | Reference only: how each House district's residents divide among boroughs, for anyone estimating where district-level ballots came from |

`population_share` is the share of the precinct's (or district's) 2020 Census
population living in that borough or census area.

## Sources

| Input | URL | SHA-256 of the download | Retrieved |
| --- | --- | --- | --- |
| 2023 election precincts (Division of Elections, District Maps page) | <https://www.elections.alaska.gov/doc/maps/AK_Election_Precincts_2023Final.zip> | `703f383acfcc2391086ea474e4b4a406cfab3ae5cdb99aebb1308e76fdea7962` | 2026-10-23 |
| 2020 Census blocks with population (TIGER/Line) | <https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_02_tabblock20.zip> | `6fe96f3a8bbaf78945ee3a3897264ed74f1b72bc0cb46ffd434039729dccf493` | 2026-10-23 |
| 2023 counties and equivalents, for names (TIGER/Line) | <https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip> | `692e12c30c83adcaabdbac0d3954fafa55e1c89a24b36d95e72e02dff938652e` | 2026-10-23 |

Census TIGER/Line files are U.S. Government works. The licence terms of the
Division of Elections precinct layer are **UNKNOWN**; only derived precinct
identifiers and population counts are stored here, not its geometry.

## Method

`election/tools/alaska_crosswalk.py` regenerates both files. Each populated 2020 Census
block is placed in the 2023 precinct containing its internal point, measured in
Alaska Albers (EPSG:3338), and counted toward the borough or census area in the
block's own county code. Population is used instead of land area because Alaska
precincts cover large uninhabited areas and water: by area, Levelock appeared
26% Kodiak Island Borough, almost all of it open water.

Checks on the 2026-10-23 build:

- All 733,391 people counted by the 2020 Census are placed; none are left over.
- All 402 precincts and all 30 boroughs and census areas appear.
- Precinct IDs (`DD-NNN`) match the `Pct_Id` column of the 2024 general and
  2026 primary precinct results exactly. Join on the ID: names differ in
  punctuation (`Pelican/Elfin Cove` here, `Pelican-Elfin Cove` in results).

Five precincts cross a borough line. Only South Tongass does so by more than 2%:

| Precinct | Largest | Other |
| --- | --- | --- |
| 01-670 South Tongass | Ketchikan Gateway 96.8% | Prince of Wales-Hyder 3.2% |
| 36-118 Mentasta | Copper River 98.3% | Southeast Fairbanks 1.7% |
| 36-042 Circle | Yukon-Koyukuk 99.3% | Southeast Fairbanks 0.7% |
| 36-062 Eagle | Southeast Fairbanks 99.5% | Yukon-Koyukuk 0.5% |
| 29-545 Valdez No. 3 | Chugach 99.5% | Copper River 0.5% |

## Limits

- Borough totals are exact only for election-day precinct votes. Absentee,
  early and questioned ballots are reported per House district, so the watcher
  leaves them in the statewide totals and never apportions them. Any borough
  share of them would be an estimate from population, which is not turnout,
  spread in rural districts across up to seven census areas.
- The watcher counts each split precinct wholly in its main borough, so counts
  stay exact integers; South Tongass (3.2% Prince of Wales-Hyder) is the only
  one where this moves more than 2% of a precinct's residents.
- Precinct boundaries are those in force from 2023. Rebuild this file if the
  Division of Elections redraws precincts.
