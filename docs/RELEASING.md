<!--
SPDX-FileCopyrightText: 2025 Donovan Worrell

SPDX-License-Identifier: MIT
Catalyst-File-ID: urn:uuid:fbc815f2-ea0e-4823-bb1d-32a0ac71abd9
-->

# Releasing

This repository follows [Catylist ADR-0013](https://github.com/dlworrell/Catylist/blob/main/docs/adr/ADR-0013-daily-fitment-and-release-branches.md):
work moves through three protected branches, and a release is published only
after the owner approves it.

## Branches

| Branch | Holds | Changes by |
|---|---|---|
| `dev-daily` | Daily work: fixes and current issues | Pull request from a topic branch (`fix/<issue>-<slug>`, `feat/<slug>`, `docs/<slug>`) cut from `dev-daily` |
| `fitment` | Promoted work while AEMS reviews and tests it | Promotion pull request `dev-daily` → `fitment`, opened by `catylist-ring-0` |
| `main` (default) | Released work only | Release candidate pull request `fitment` → `main`, opened by `catylist-ring-0` |

All three refuse deletion, force pushes and direct pushes, and accept only
signed commits and merge commits (Catylist ADR-0006). The `fitment/source`
check fails a pull request into `fitment` whose head is not `dev-daily`, and
one into `main` whose head is not `fitment`.

## How a change reaches a release

1. **Merge into `dev-daily`.** Open a pull request from your topic branch; CI
   and ring-0 run on it.
2. **Promotion.** When `dev-daily` is ahead of `fitment` and green, the AEMS
   fitment cycle (every 30 minutes) opens the promotion pull request. The
   owner merges it.
3. **Fitment review.** AEMS reviews what reached `fitment` and files each
   finding as an issue labelled `fitment` and its severity. Fix it on
   `fix/<issue>-<slug>` into `dev-daily` with `Fixes #<issue>` in the commit
   message; the issue closes when that commit reaches `fitment`. The
   `fitment` check fails while any test check fails or a `blocker` or `major`
   fitment issue is open.
4. **Release candidate.** When the `fitment` check passes, AEMS opens the
   release candidate pull request `fitment` → `main`. The owner merges it.
5. **Publish.** The owner runs the **Release** workflow
   (`.github/workflows/release.yml`) on `main`:
   [Actions → Release → Run workflow](https://github.com/dlworrell/code-noodling/actions/workflows/release.yml),
   branch `main`, select the project ID (`prime-number-suite`, `crypto-pipeline`,
   or `election-analysis`), and enter the version (no leading `v`;
   `0.1.0-rc.1` makes a pre-release). The workflow builds an attested source
   archive and records the exact tested commit and archive SHA-256.
6. **Approve.** The `publish` job waits at the `release` environment. Open the
   run, choose **Review deployments**, tick `release` and **Approve and
   deploy**. The workflow rechecks main and protection before atomically
   creating `<project>/v<version>` on the tested commit, then attaches the
   verified archive and scoped notes to the GitHub release. Existing
   repository-wide tags such as `v0.1.0` stay immutable and unchanged.

The workflow refuses, before anything is published, when it is run from any
branch but `main`, when the version is malformed, when the tag already exists
(release tags are immutable), or when any check on `main`'s head has not
passed. If main advances while approval is pending, dispatch again for the new
commit. If ref creation succeeds but release or asset publication fails,
inspect the immutable tag and the release state; do not delete, move or blindly
retry that tag via this workflow.

## What the project asset contains

The custom attachment is a deterministic archive of the selected project's
tracked source and applicable license texts, not a binary build. Election
source archives keep the tracked watcher/capture configuration, but exclude
third-party election snapshots, crosswalks and data-bearing test fixtures
and omit generated reports. Report publishing still submits separate report
updates to `dev-daily`. GitHub's automatic “Source code” archives are
repository-wide; use the named, attested attachment for the project archive.

Notes cover the project directory and shared licenses since the newest
reachable tag in that project's namespace by Git tag creation date. The
first project release instead shows the scoped history through its tested
commit; a repository-wide tag is never used as a per-project baseline.
Use `gh attestation verify <archive> --repo dlworrell/code-noodling` to
verify provenance for the downloaded attachment.

## One-time setup: the `release` environment

The approval in step 6 depends on the `release` environment at
<https://github.com/dlworrell/code-noodling/settings/environments>. It should have:

- **Required reviewers:** the repository owner;
- **Prevent self-review:** off (a single-owner repository has no one else to
  approve);
- **Deployment branches and tags:** selected branches, `main` only.

AEMS's **Apply Catylist Rulesets** workflow creates and checks this setup for
every repository in AEMS `config/fitment.json`. Without the environment,
`publish` runs without waiting for approval, so check it before the first
release.

## One-time setup: immutable project tags

Before first use, extend the existing active repository tag ruleset
`protect-release-tags` (ID 25747644) without removing its existing
`refs/tags/v*` include. Add these exact includes:

- `refs/tags/prime-number-suite/v*`
- `refs/tags/crypto-pipeline/v*`
- `refs/tags/election-analysis/v*`

Keep no ref exclusions, no bypass actors, and update, deletion, and
non-fast-forward restrictions. The workflow fails closed before packaging
and again after approval if it cannot verify these conditions for the
selected project. The stock Actions token may not see `bypass_actors`; GitHub
documents that the API hides this field from identities without write
access to the ruleset. `RELEASE_RULESET_READ_TOKEN` is optional input to the
read-only check; if supplied, use an explicitly authorized, repository-scoped
credential, and do not add it merely to attempt to bypass a failed check.
If the field is unavailable, no release proceeds. The setup is separate from
releasing, does not change legacy tags, and does not itself authorize a tag.
