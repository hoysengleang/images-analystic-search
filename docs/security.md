# Security model

This document explains the project's trust boundaries, built-in controls, and
operator responsibilities. Report vulnerabilities privately as described in
[`SECURITY.md`](../SECURITY.md).

## Current release boundary

The checked-in storage foundation is tenant-scoped, but the live `/collections`
API is still the legacy image-level API and uses one optional server-wide API
key. **Do not expose the current release as a hostile multi-tenant service.**
The `/v1` product API will bind authenticated tenants to the tenant-scoped
repositories described in the
[product specification](VISUAL_PRODUCT_SEARCH_SPEC.md).

Until then, treat one deployment as one administrative trust domain and keep it
behind an application backend or private network.

## Trust boundaries

OpenVisionSearch processes several forms of untrusted input:

| Boundary | Main risks | Controls |
| --- | --- | --- |
| HTTP clients | oversized bodies, malformed requests, unauthorized access | request-size middleware, schema validation, optional API key |
| Uploaded or encoded images | decompression bombs, invalid formats, excessive memory use | byte, MIME, format, and decoded-pixel limits; bounded batches; explicit resource cleanup |
| Remote image URLs | SSRF, redirect-to-private-host, slow or oversized responses | scheme checks, DNS/IP validation on every hop, redirect and timeout limits, streamed byte cap |
| Mounted paths | traversal, symlink escape, unintended file reads | canonical path containment and allowed-root enforcement |
| Search storage | cross-tenant reads or writes, incompatible vectors | tenant-bound repositories and mandatory tenant/model fields in the engine contract |
| Model and container downloads | compromised upstream artifacts or unexpected upgrades | documented model identity, pinned Qdrant release, bounded dependency versions |

Original images are decoded only long enough to create a query or stored vector.
They are not copied into application storage. Docker Compose mounts the merchant
image directory read-only.

## Deployment checklist

For any deployment reachable outside a developer machine:

1. Set a strong `API_KEY`; never put it in source control or a URL.
2. Put the API behind TLS and a reverse proxy with connection and request-rate
   controls.
3. Set `CORS_ALLOW_ORIGINS` to the exact trusted browser origins. CORS is not an
   authentication mechanism.
4. Keep `ALLOW_PRIVATE_URLS=false` unless every caller is fully trusted and the
   network impact is understood.
5. Do not mount secrets, the Docker socket, a home directory, or a broad host
   directory beneath `ALLOWED_IMAGE_ROOT`.
6. Keep Qdrant private to the application network unless operators explicitly
   require its ports. Configure Qdrant authentication when it crosses a trusted
   network boundary.
7. Back up the vector and metadata volumes together and test restoration.
8. Review dependency and container alerts before each release.
9. Avoid logging image payloads, signed URLs, API keys, authorization headers,
   or merchant catalogue records.

The Compose API service runs as UID 10001 with no Linux capabilities, a read-only
root filesystem, and an isolated temporary filesystem. These are defense-in-depth
controls and do not replace network isolation or authentication.

## Tenant-isolation requirements

New product APIs and background jobs must resolve one tenant before accessing
tenant-owned data. Tenant scope belongs at repository construction and in every
search-engine operation. A route-level filter alone is not sufficient.

Every tenant-isolation change needs negative tests that attempt cross-tenant
reads, updates, deletes, and vector searches. Admin credentials may perform
administrative operations but must not silently acquire a tenant scope.

## Model integrity and compatibility

Every vector records its embedding provider, model identifier, revision,
dimension, and normalization behavior. Search must reject incompatible model
versions rather than compare them. Operators should review the default model's
license and model card and should mirror or pin artifacts when their supply-chain
policy requires fully reproducible downloads.

## Out of scope guarantees

The project does not currently claim certification, formal sandboxing of image
decoders, a security service-level agreement, or protection against a malicious
host administrator. Resource limits reduce risk but are not a substitute for
container-orchestrator CPU and memory limits.
