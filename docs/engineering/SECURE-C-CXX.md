# Secure C/C++ Profile

Status: Adopted
Repository: dlworrell/code-noodling
Inherited Standard: AES-SEC-001

## Policy

This repository inherits `AES-SEC-001: Secure C and C++ Coding Rules` from the Atarix Engineering Standard.

No new C or C++ code may be accepted unless it follows the AES secure-coding profile.

## Required Local Behavior

If this repository contains or later accepts C/C++ code, that code must:

- avoid banned unsafe interfaces;
- carry explicit lengths for external buffers;
- validate serialized lengths before use;
- check allocation and copy-size arithmetic for overflow;
- avoid signed-to-unsigned length conversion until bounds are proven;
- use RAII and smart ownership for C++ resource management;
- isolate unsafe code behind reviewed interfaces;
- compile cleanly under the repository warning profile;
- run sanitizer tests when supported by the target platform;
- include fuzz coverage for parsers and external-input handlers when applicable;
- document all approved exceptions as waivers.

## Unsafe Code

Unsafe operations are allowed only when they are required by the platform, ABI, hardware interface, allocator, parser, or other low-level boundary. Unsafe code must be isolated and documented with the invariant that makes the operation safe enough.

## Waivers

A waiver is required for any violation of an AES `MUST` or `BANNED` rule. Waivers must identify the rule, file path, reason, safety invariant, owner, review date, and supporting test evidence.

## Ratchet Rule

Existing legacy violations may be baselined during adoption. New violations should block merge unless a waiver is recorded.
