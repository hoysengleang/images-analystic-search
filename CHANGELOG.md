# Changelog

All notable changes to this project will be documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Multi-tenant SQLite foundation and tenant-scoped repositories.
- Search-engine and embedding-provider contracts with model-version separation.
- SQLite-backed native exact search with tenant/model cache partitioning,
  product filters, durable vectors, and deterministic cosine ranking.
- Contributor, security, support, issue, and pull-request guidance.
- Automated dependency updates and CodeQL security analysis.
- Query-side region search: a search may carry a crop rectangle, or ask the
  server to detect the objects in the query image and search each one. Indexed
  images are unchanged, and the detector is off by default. See
  [ADR 0007](docs/adr/0007-query-side-region-search.md).
- An ONNX Runtime embedding provider, so a server can run search without
  PyTorch. `scripts/export_onnx.py` exports the encoders and refuses to
  finish unless they reproduce the PyTorch model on real inputs.

### Changed

- The interactive API reference at `/docs` is rendered by Scalar instead of
  Swagger UI. ReDoc is unchanged at `/redoc`.
- Development dependencies are separated from production dependencies.
- Route response models, tenant guards, embedding configuration, and Qdrant
  filter compilation now have single, explicit owners.
- The API container runs as an unprivileged user with a read-only root filesystem.
- Qdrant is pinned to a specific release for reproducible Compose deployments.
- Qdrant is private to the Compose network instead of publishing its ports on
  the host by default.
- The default container and CI install official CPU-only PyTorch wheels on
  amd64 and arm64 instead of pulling unused CUDA libraries.
- Compose stores the collection registry in the `app_state` volume instead of
  the writable `./data` bind mount; existing deployments must migrate
  `./data/collections.json` as described in the deployment guide.

### Security

- Hardened image loading, URL validation, tenant isolation, API-key handling, and
  resource cleanup.
- Unsupported filter value types now return `UNSUPPORTED_FILTER` instead of
  leaking an internal `TypeError`.
- EXIF-orientation intermediates are closed immediately after image decoding.

[Unreleased]: https://github.com/hoysengleang/images-analystic-search/compare/main...HEAD
