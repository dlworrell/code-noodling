"""Build the Alaska precinct -> borough/census area crosswalk.

Alaska has no counties. Its county equivalents are 19 organized boroughs and
11 census areas of the Unorganized Borough. Election results are reported by
precinct and House district, so this script works out which borough or census
area each precinct's voters live in.

Method: every populated 2020 Census block is placed in the 2023 election
precinct containing the block's internal point, and counted toward the borough
or census area in the block's own county code. A precinct's share of a borough
is its share of population, not of land, because Alaska precincts span large
unpopulated areas and water.

This is a one-off, offline tool. It is not run by CI and needs packages the
analyzer does not: pyshp, shapely 2 and pyproj.

    python election/tools/alaska_crosswalk.py \\
        --precincts Precincts.shp \\
        --blocks tl_2020_02_tabblock20.shp \\
        --counties tl_2023_us_county.shp \\
        --output election/data/crosswalks/alaska
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy
import shapefile
import shapely
from pyproj import Transformer
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.validation import make_valid

ALASKA_FIPS = "02"
# Alaska Albers equal area, metres.
TARGET_CRS = 3338
# Blocks whose internal point falls just outside every precinct (coastline
# differences between the two sources) go to the nearest precinct within this
# distance; anything further is reported as unplaced.
NEAREST_LIMIT_M = 2000


def _projector(source_crs: int):
    transformer = Transformer.from_crs(source_crs, TARGET_CRS, always_xy=True)

    def project(coords: numpy.ndarray) -> numpy.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return numpy.column_stack([x, y])

    return transformer, project


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precincts", required=True, type=Path)
    parser.add_argument("--blocks", required=True, type=Path)
    parser.add_argument("--counties", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    # The state's precinct layer is Web Mercator; Census layers are NAD83.
    _, from_mercator = _projector(3857)
    from_nad83, _ = _projector(4269)

    names = {
        rec["COUNTYFP"]: rec["NAMELSAD"]
        for rec in shapefile.Reader(str(args.counties)).iterRecords(fields=["STATEFP", "COUNTYFP", "NAMELSAD"])
        if rec["STATEFP"] == ALASKA_FIPS
    }

    precincts = []
    for item in shapefile.Reader(str(args.precincts)).iterShapeRecords():
        record = item.record.as_dict()
        geometry = make_valid(shapely.transform(shape(item.shape.__geo_interface__), from_mercator))
        name = " ".join(record["Precinct_N"].split())
        precincts.append(
            {
                "id": name[:6],
                "name": name,
                "house": record["HouseDistr"].strip(),
                "senate": record["SenateDist"].strip(),
                "geometry": geometry,
            }
        )
    tree = STRtree([p["geometry"] for p in precincts])

    population: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    unplaced: collections.Counter = collections.Counter()
    fields = ["COUNTYFP20", "POP20", "INTPTLAT20", "INTPTLON20"]
    for rec in shapefile.Reader(str(args.blocks)).iterRecords(fields=fields):
        people = int(rec["POP20"])
        if not people:
            continue
        point = shapely.Point(from_nad83.transform(float(rec["INTPTLON20"]), float(rec["INTPTLAT20"])))
        hits = [i for i in tree.query(point) if precincts[i]["geometry"].covers(point)]
        if not hits:
            nearest = int(tree.nearest(point))
            if precincts[nearest]["geometry"].distance(point) > NEAREST_LIMIT_M:
                unplaced[rec["COUNTYFP20"]] += people
                continue
            hits = [nearest]
        population[hits[0]][rec["COUNTYFP20"]] += people

    args.output.mkdir(parents=True, exist_ok=True)
    district: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with (args.output / "precinct-borough.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["precinct_id", "precinct_name", "house_district", "senate_district",
             "borough_fips", "borough", "population_2020", "population_share"]
        )
        for index, precinct in sorted(enumerate(precincts), key=lambda item: item[1]["id"]):
            counts = population[index]
            total = sum(counts.values())
            for county, people in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                writer.writerow(
                    [precinct["id"], precinct["name"], precinct["house"], precinct["senate"],
                     ALASKA_FIPS + county, names[county], people, f"{people / total:.6f}"]
                )
                district[precinct["house"]][county] += people
    with (args.output / "house-district-borough.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["house_district", "borough_fips", "borough", "population_2020", "population_share"])
        for house in sorted(district):
            counts = district[house]
            total = sum(counts.values())
            for county, people in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                writer.writerow([house, ALASKA_FIPS + county, names[county], people, f"{people / total:.6f}"])

    placed = sum(sum(c.values()) for c in population.values())
    print(f"placed {placed} people in {len(precincts)} precincts; unplaced {sum(unplaced.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
