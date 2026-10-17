# Hardware execution security: no public lab runners

## Decision

Do not register the K80/DL380p lab machine as a runner for this public
repository, including runners labeled `k80`, `kepler`, or `cuda11`. Keep
Settings → Actions → Runners empty. Do not grant this repository access through
an organization runner group either.

A fork pull request can add a new workflow that selects a registered runner.
Manual triggers, job conditions, labels, environment approvals, and requiring
approval for external contributors in the existing workflows do not prevent
that new workflow from selecting the machine. Contributor approval remains a
useful additional control, but is not the lab isolation boundary.

The public hardware workflow now runs only on GitHub-hosted infrastructure and
fails explicitly when requested. It does not build on the lab, dispatch work to
it, or claim hardware validation. Portable CI and CUDA compile-only checks
continue independently. `scripts/ci/check_public_runners.py` and its CI workflow
check for source regressions, including added workflows, custom labels, groups,
and dynamic selectors. A fork can change or omit this check, so it is not an
authorization control and does not make public runner enrollment safe.

## Current operating procedure

Hardware validation is an explicit local operator action. Review the source and
build/test scripts at a full commit SHA on the protected default branch. Record
that SHA, toolchain versions, device inventory, commands, and results with the
review evidence. Use a clean checkout of that reviewed SHA. Do not run a PR head,
fork ref, PR artifact, downloaded executable, or an unreviewed build script on
the lab machine. Do not start an Actions runner service for this repository.

If a runner is ever found enrolled, stop its service and remove its registration
before resuming lab use. Investigate jobs already executed; removal alone cannot
prove that the host or its credentials were not compromised.

## Future automation boundary

Before enabling automated hardware jobs, use a separate private operations
repository with access limited to trusted operators. Public repositories must
have no runner access, no dispatch credential, and no automated event/artifact
bridge into that repository. The private workflow must accept only reviewed,
immutable commits from the expected upstream repository and protected branch;
never execute arbitrary refs, commands, or public PR artifacts. Restrict runner
access to that private repository and reviewed workflows, isolate the lab and
credentials, and require a separate security review before enrollment.

Until that private control boundary is implemented and verified, use local
operator validation only. This change does not create or enroll a private runner.

## Public/private policy setting

The policy is `public repository => no self-hosted runner access`. These
repositories are personally owned; there is no organization runner group to
configure here. For a future organization deployment, use a runner group with
private or selected-repository access, leave public access disabled, and disable
repository-level self-hosted runners on public repositories. Those are GitHub
access controls; an AEMS/Catylist public/private feature should audit their live
state, not rely on a flag inside a workflow from the PR under review.

## Verification

- Confirm the repository runner inventory is empty in GitHub settings.
- Confirm organization runner groups, if introduced later, do not allow this
  public repository.
- Run `python scripts/ci/check_public_runners.py` and
  `python -m unittest discover -s tests/ci -p 'test_public_runners.py'`.
- A manual request for the public hardware workflow must fail on a GitHub-hosted
  runner with the local-validation message; it must not queue for the lab.

GitHub guidance:
https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions
https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access

## Local K80 validation

Prerequisites: NVIDIA R470, CUDA 11.4, GCC 10 or older, CMake 3.20+, Python 3,
and the expected GPU devices. After reviewing and checking out the full commit
SHA, run from the repository root:

```bash
nvidia-smi
nvcc --version
cmake -S prime-number-suite -B build-k80 \
  -DBUILD_CUDA_SIEVE_MGPU=ON -DBUILD_PHYSX_DICE=OFF -DBUILD_TESTING=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=37 \
  -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets
cmake --build build-k80 --target cuda_sieve_mgpu ose_cpu --parallel
./build-k80/cuda_sieve_mgpu 10000000 --gpus 8 --json gpu.json
./build-k80/ose_cpu 1 10000000 --json cpu.json --no-list
python3 - <<'COMPARE'
import json
from pathlib import Path

gpu = json.loads(Path("gpu.json").read_text())["primes"]
cpu = json.loads(Path("cpu.json").read_text())["primes"]
if gpu != cpu:
    raise SystemExit("GPU and CPU prime lists differ")
print("GPU and CPU prime lists match")
COMPARE
```

Choose the GPU count from the actual inventory. A local result is hardware
evidence for the recorded SHA, not a public CI success or a bypass of review.
