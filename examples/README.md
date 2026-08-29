# Examples

Every example does the same four things against a server on `http://localhost:8000`:

1. create a collection,
2. index images from a URL,
3. search by image, by text, and by both together,
4. read the results.

| Language | File |
| --- | --- |
| curl | [`curl/quickstart.sh`](curl/quickstart.sh) |
| curl (every endpoint) | [`curl/full-api-tour.sh`](curl/full-api-tour.sh) |
| Python | [`python/quickstart.py`](python/quickstart.py) |
| Python (bulk) | [`python/bulk_index_from_database.py`](python/bulk_index_from_database.py) |
| JavaScript | [`javascript/quickstart.mjs`](javascript/quickstart.mjs) |
| PHP | [`php/quickstart.php`](php/quickstart.php) |

`full-api-tour.sh` walks all 15 endpoints in the order you would use them, with notes on the parts that surprise people.

Set `OVS_API_KEY` if the server runs with `API_KEY` configured.
