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
   branch `main`, version such as `0.1.0` (no leading `v`; `0.1.0-rc.1` makes
   a pre-release).
6. **Approve.** The `publish` job waits at the `release` environment. Open the
   run, choose **Review deployments**, tick `release` and **Approve and
   deploy**. Only then is the `v<version>` tag created on the tested commit,
   with a GitHub release and generated notes.

The workflow refuses, before anything is published, when it is run from any
branch but `main`, when the version is malformed, when the tag already exists
(release tags are immutable), or when any check on `main`'s head has not
passed.

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
