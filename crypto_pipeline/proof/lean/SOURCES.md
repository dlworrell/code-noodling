<!--
SPDX-FileCopyrightText: 2026 Donovan Worrell

SPDX-License-Identifier: Apache-2.0
Catalyst-File-ID: urn:uuid:4ed5dbfe-57af-4891-93af-9eb48a45d8cf
-->

# Sources the Lean proofs are compiled against

This directory's own sources (`*.lean`, `lakefile.toml`, `lean-toolchain`) are
committed. This file records the exact external sources the proofs are built
with, so a build can be reproduced even if a GitHub repository, tag or
release disappears.

`lake-manifest.json` (committed next to this file) pins every Lake
dependency to a commit. `lake build` uses those commits; `lake update` would
re-resolve them, so run it only to move to a new Mathlib on purpose, and
update this table in the same change.

## Pinned sources

All 10 commits below were confirmed present in the Software Heritage archive
(archive.softwareheritage.org, `GET /api/1/revision/<sha>/`) on 2026-11-23.
Software Heritage keeps public source code independently of GitHub; the
SWHIDs below identify these exact commits there.

| Component | Repository | Ref | Commit (SWHID) |
|---|---|---|---|
| Lean 4 toolchain | github.com/leanprover/lean4 | `v4.15.0` (`lean-toolchain`) | `swh:1:rev:11651562caae0a0b3973811508db2ab8903d3854` |
| mathlib | github.com/leanprover-community/mathlib4 | `v4.15.0` | `swh:1:rev:9837ca9d65d9de6fad1ef4381750ca688774e608` |
| batteries | github.com/leanprover-community/batteries | `v4.15.0` | `swh:1:rev:e8dc5fc16c625fc4fe08f42d625523275ddbbb4b` |
| aesop | github.com/leanprover-community/aesop | `v4.15.0` | `swh:1:rev:2689851f387bb2cef351e6825fe94a56a304ca13` |
| Qq | github.com/leanprover-community/quote4 | `v4.15.0` | `swh:1:rev:f0c584bcb14c5adfb53079781eeea75b26ebbd32` |
| plausible | github.com/leanprover-community/plausible | `v4.15.0` | `swh:1:rev:2c57364ef83406ea86d0f78ce3e342079a2fece5` |
| importGraph | github.com/leanprover-community/import-graph | `v4.15.0` | `swh:1:rev:9a0b533c2fbd6195df067630be18e11e4349051c` |
| proofwidgets | github.com/leanprover-community/ProofWidgets4 | `v0.0.48` | `swh:1:rev:2b000e02d50394af68cfb4770a291113d94801b5` |
| LeanSearchClient | github.com/leanprover-community/LeanSearchClient | `main` (moving) | `swh:1:rev:003ff459cdd85de551f4dcf95cdfeefe10f20531` |
| Cli | github.com/leanprover/lean4-cli | `main` (moving) | `swh:1:rev:0c8ea32a15a4f74143e4e1e107ba2c412adb90fd` |

The two `main` refs move upstream; the commit pins in `lake-manifest.json`
(and above) are what keep the build fixed.

Lean toolchain binary used for local checks: the official release asset
`lean-4.15.0-linux.tar.zst` from github.com/leanprover/lean4/releases/tag/v4.15.0,
SHA-256 `af71a2569a9f68337de2434829b3008cd8e32c436e9cb6bd8c84a2c2ba3585c9`
(computed 2026-11-23); `lean --githash` reports the commit above. CI installs
the toolchain through elan, whose installer the workflow pins by SHA-256.

Not recorded here: the Mathlib build cache (`lake exe cache get`). It holds
compiled outputs only and can be rebuilt from the sources above with
`lake build` (hours on one machine, instead of minutes).

## Recovering a source from Software Heritage

If a repository or commit is gone from GitHub, its archived copy can be
retrieved through the Software Heritage vault (API documented at
docs.softwareheritage.org, "Vault" endpoints):

```sh
SWHID=swh:1:rev:9837ca9d65d9de6fad1ef4381750ca688774e608   # e.g. mathlib
curl -X POST https://archive.softwareheritage.org/api/1/vault/git-bare/$SWHID/
curl https://archive.softwareheritage.org/api/1/vault/git-bare/$SWHID/   # until "status": "done"
curl -LOJ https://archive.softwareheritage.org/api/1/vault/git-bare/$SWHID/raw/
tar -xf swh_1_rev_*.git.tar        # unpacks to the directory swh:1:rev:<sha>.git
git clone swh:1:rev:*.git mathlib
```

Then point the package's `url` in `lake-manifest.json` (or a local
`packagesDir` checkout) at the recovered repository; the `rev` stays the
same.

## Re-checking that the commits are archived

```sh
python3 - <<'PY'
import json, urllib.request
for p in json.load(open("lake-manifest.json"))["packages"]:
    u = f"https://archive.softwareheritage.org/api/1/revision/{p['rev']}/"
    try:
        urllib.request.urlopen(u, timeout=30); print("archived", p["name"])
    except Exception as e:
        print("MISSING", p["name"], e)
PY
```
