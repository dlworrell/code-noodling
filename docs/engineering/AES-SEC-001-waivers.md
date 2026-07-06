# AES-SEC-001 Waiver Log

Status: Active
Owner: code-noodling
Standard: AES-SEC-001

## Purpose

This file records approved exceptions to `AES-SEC-001: Secure C and C++ Coding Rules` for this repository.

A waiver is required when project-owned code intentionally violates an AES `MUST` or `BANNED` rule.

## Current Waivers

No waivers are currently approved for this repository.

## Waiver Template

```text
Waiver ID:
Rule:
File:
Line or Symbol:
Reason:
Safety Invariant:
Compensating Controls:
Test Evidence:
Owner:
Review Date:
Expiration or Recheck Condition:
```

## Rules

- A waiver is not permission to spread unsafe patterns.
- A waiver must be local, specific, and reviewable.
- A waiver must identify the invariant that makes the exception safe enough.
- A waiver should be removed when the underlying code can be corrected.
