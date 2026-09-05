# Security policy

Tenant isolation and safe handling of untrusted images are the project's highest
priorities. Please report security problems privately so maintainers can protect
users before details are public.

## Supported versions

Security fixes are made on the `main` branch and released in the newest version.
Pre-release versions may change quickly and older releases may not receive fixes.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `main` | Yes |
| Older releases | No |

## Report a vulnerability

Use GitHub's
[private vulnerability reporting](https://github.com/hoysengleang/images-analystic-search/security/advisories/new).
Do not include vulnerability details in a public issue, discussion, or pull request.

Include, where possible:

- the affected version or commit;
- the deployment configuration and search backend;
- reproduction steps or a minimal proof of concept;
- the impact, especially any cross-tenant access or server-side request forgery;
- suggested mitigations, if known.

Maintainers aim to acknowledge a complete report within three business days. They
will coordinate validation, remediation, release, and disclosure with the reporter.
Response time may vary because this is a volunteer-maintained project.

There is currently no paid bug-bounty program. Good-faith research that avoids
privacy violations, data destruction, denial of service, and access beyond what is
needed to demonstrate the issue is welcome.

## Security boundaries

The threat model and operator responsibilities are documented in
[`docs/security.md`](docs/security.md). In particular, production operators should
configure authentication, restrict CORS, keep private-URL access disabled, and put
the API behind TLS and appropriate network controls.
