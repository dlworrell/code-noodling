# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:6b03e726-cd82-4206-864f-7257f03b0b80

"""Archival format oracles and malformed-input checks (not calibration)."""
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from election_reporting.archives import KING_COLUMNS, MEDSL_COLUMNS, king_snapshot, alaska_medsl
from election_reporting.ingest import read_snapshot
from election_reporting.model import ElectionReportError
from election_reporting.model import ElectionConfig,SourceConfig
from election_reporting.analysis import analyze_election
from election_reporting.publish import analysis_payload,build_edom
from election_reporting.watch.alaska import Crosswalk
from election_reporting.watch.sources import Place


def encoded_rows(columns, rows, encoding="utf-8"):
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode(encoding)


class ArchiveTests(unittest.TestCase):
    def test_king_identity_units_encoding_and_malformed_rows(self):
        a=dict(zip(KING_COLUMNS,("1","1","City","","City A","Mayor","100","200","1","René","60")))
        b=dict(a,**{"District Name":"City B","Votes":"20"})
        result=king_snapshot(encoded_rows(KING_COLUMNS,[a,b],"cp1252"))
        self.assertEqual(result["encoding"],"cp1252")
        self.assertEqual(len(result["contests"]),2)
        self.assertEqual([r["district_ballots"] for r in result["contests"].values()],[100,100])
        self.assertTrue(all(r["contest_undervotes"] is None and not r["whole_contest_coverage"] for r in result["contests"].values()))
        for invalid in ([a,a],[dict(a,Votes="-1")],[dict(a,Votes="1.5")],[dict(a,Votes="9"*12)]):
            with self.assertRaises(ElectionReportError):king_snapshot(encoded_rows(KING_COLUMNS,invalid))
        with self.assertRaises(ElectionReportError):king_snapshot(b"GEMS Contest ID,GEMS Contest ID\n1,1\n")
        with self.assertRaises(ElectionReportError):king_snapshot(b"\x00")

    def test_medsl_all_modes_once_and_district_rows_never_become_borough_votes(self):
        def row(precinct,mode,candidate,votes):
            return dict(zip(MEDSL_COLUMNS,(precinct,"US HOUSE",candidate,"STATEWIDE",mode,str(votes),"DEM","2024","ALASKA","GEN","1")))
        rows=[row("001-600","ELECTION DAY","A",10),row("001-600","ELECTION DAY","B",5),
              row("001-600","ELECTION DAY","UNDERVOTES",2),row("District 1 - Absentee","ABSENTEE","A",7),
              row("District 1 - Absentee","ABSENTEE","A",0),row("District 1 - Early Voting","ELECTION DAY","B",3),
              row("HD99 Fed Overseas Absentee","FEDERAL OVERSEAS","A",1),
              row("District 1 - Question","QUESTION","B",2),row("District 1 - Early","EARLY VOTING","B",4)]
        crosswalk=Crosswalk({"01-600":"Example Borough"},("Example Borough",),("01-600",))
        places=[Place("alaska","Alaska","state",""),Place("example","Example Borough","county","")]
        outputs,evidence=alaska_medsl(encoded_rows(MEDSL_COLUMNS,rows),crosswalk,places)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"2026-10-25.csv"
            path.write_text(outputs["alaska"]);state=next(iter(read_snapshot(path,"alaska").contests.values()))
            path.write_text(outputs["example"]);borough=next(iter(read_snapshot(path,"example").contests.values()))
            analysis=analyze_election(ElectionConfig("Synthetic borough",(SourceConfig("example",Path(directory),"Example Borough","county",1),),frozenset()))
            for rendered in (str(analysis_payload(analysis)),str(build_edom(analysis))):
                self.assertIn("approximate borough assignment",rendered)
                self.assertIn("geometry_year",rendered)
                self.assertIn("2023",rendered)
                self.assertIn("excluded_categories",rendered)
            self.assertTrue(all(not r.controlling and not r.forecast_eligible for r in analysis.races))
        self.assertEqual({c.name:c.votes for c in state.choices},{"A":18,"B":14})
        self.assertEqual(state.ballots,34)
        self.assertEqual({c.name:c.votes for c in borough.choices},{"A":10,"B":5})
        self.assertEqual(borough.ballots,17)
        self.assertTrue(borough.metadata["geography"]["approximate"])
        self.assertFalse(borough.metadata["geography"]["whole_contest_coverage"])
        self.assertIn("district-level rows mislabeled ELECTION DAY",borough.metadata["geography"]["excluded_categories"])
        self.assertEqual(evidence["zero_filled_repetitions"],1)
        self.assertEqual(evidence["district_rows_mislabeled_election_day"],["District 1 - Early Voting"])
        with self.assertRaisesRegex(ElectionReportError,"repeated positive"):
            alaska_medsl(encoded_rows(MEDSL_COLUMNS,rows+[rows[0]]),crosswalk,places)
        with self.assertRaisesRegex(ElectionReportError,"missing from crosswalk"):
            alaska_medsl(encoded_rows(MEDSL_COLUMNS,[row("002-999","ELECTION DAY","A",1)]),crosswalk,places)

    def test_frozen_king_archives_have_hash_bound_descriptive_coverage(self):
        fixtures=Path(__file__).parent/"fixtures"
        manifest=json.loads((fixtures/"king-archives-manifest.json").read_text())
        self.assertEqual(hashlib.sha256((fixtures/"king-archives.zip").read_bytes()).hexdigest(),manifest["archive_sha256"])
        with zipfile.ZipFile(fixtures/"king-archives.zip") as archive:
            self.assertEqual(set(archive.namelist()),{r["file"] for r in manifest["files"]})
            self.assertEqual(len(manifest["files"]),63)
            for spec in manifest["files"]:
                info=archive.getinfo(spec["file"])
                self.assertLess(info.file_size,8_000_000)
                raw=archive.read(info)
                self.assertEqual(hashlib.sha256(raw).hexdigest(),spec["sha256"])
                snapshot=king_snapshot(raw)
                self.assertEqual(len(snapshot["contests"]),spec["contests"])
                self.assertEqual(snapshot["encoding"],spec["encoding"])
                self.assertFalse(spec["eligible_for_forecast_validation"])


if __name__ == "__main__":
    unittest.main()
