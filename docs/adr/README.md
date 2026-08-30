# Architecture decision records

Section 9 of the specification asks that any deviation from it, and any choice
it leaves open, be recorded here. One file per decision, numbered, never
rewritten — supersede instead.

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-evolve-the-existing-repository.md) | Evolve this repository rather than restart | Accepted |
| [0002](0002-sqlite-migrations-without-a-library.md) | Plain SQL migrations, no migration library | Accepted |
| [0003](0003-tenant-scope-in-repository-construction.md) | Tenant scope is bound at repository construction | Accepted |
| [0004](0004-api-key-storage-format.md) | API keys stored as SHA-256 hashes | Accepted |
| [0005](0005-vector-persistence-format.md) | Vectors persisted as little-endian float32 blobs | Accepted |
| [0006](0006-lazy-native-engine-cache.md) | Native vectors load lazily per tenant and model | Accepted |
