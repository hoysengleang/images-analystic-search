# 0004 — API keys stored as SHA-256 hashes

**Status:** Accepted · 2026-08-29 · Resolves an open decision from Section 26

## Context

Section 26 leaves the API-key creation and storage format open. Section 15
requires constant-time comparison and forbids logging credentials.

## Decision

Keys are `ovs_` plus 32 bytes from `secrets.token_urlsafe`. Only
`sha256(key)` is stored, in a unique column. Verification hashes the presented
key and looks it up, then confirms with `hmac.compare_digest`. The plaintext is
returned exactly once, by the call that creates it.

## Rationale

A stolen database yields no working credentials. A password KDF such as bcrypt
is unnecessary here: these keys are 256 bits of machine-generated entropy, not
human-chosen secrets, so there is nothing to brute-force and the per-request
cost of a slow KDF would be pure overhead.

## Consequences

- A lost key cannot be recovered, only replaced.
- `key_hash` is in the logger's redaction list alongside `api_key`.
- If keys ever become user-chosen, this decision must be revisited and a real
  KDF adopted.
