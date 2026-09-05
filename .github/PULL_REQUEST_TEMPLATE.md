## What changed

<!-- Describe the user-visible result and why this change is needed. -->

## Risk

<!-- What could break? Call out tenant isolation, model compatibility, migrations,
image/URL handling, persistence, and API compatibility when relevant. -->

## Verification

- [ ] `ruff check app tests examples benchmarks`
- [ ] `ruff format --check app tests examples benchmarks`
- [ ] `python -m pytest`
- [ ] Documentation and examples are updated where behavior changed
- [ ] No original images, credentials, model weights, or generated runtime data are committed
- [ ] Tenant-scoped storage and model-version boundaries remain enforced

## Related issue

<!-- Use "Closes #123" when this should close an issue. -->
