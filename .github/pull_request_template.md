# Pull Request

## Summary
Describe the change and why it is needed.

## AES-DEV-001
- [ ] Governing architecture, specification, or ADR is linked or updated.
- [ ] Documentation is updated in this change series, or the reason it is not required is stated.
- [ ] Tests are included, or a test rationale is provided.
- [ ] Interface and versioning impact is addressed.
- [ ] Failure modes, recovery, observability, and diagnostics are addressed where applicable.
- [ ] The change is one logical, reviewable unit suitable for revert and `git bisect`.

## AES-SEC-001
- [ ] Trust boundaries and authority crossings are identified where applicable.
- [ ] External lengths, indices, serialized values, allocation arithmetic, and signed conversions are validated.
- [ ] Banned unsafe interfaces are not introduced.
- [ ] Unsafe operations are isolated and their safety invariant is documented.
- [ ] Static analysis, warning-clean build, sanitizer, and fuzz evidence is supplied where applicable.
- [ ] Custom cryptography is not introduced.
- [ ] Any exception is recorded in the repository waiver log.

## Evidence
List tests, workflow runs, benchmarks, documentation, ADRs, security review, and recovery evidence.