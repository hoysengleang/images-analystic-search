"""Backfill an existing product table into OpenVisionSearch.

The pattern is the same whether your images sit on a CDN, in S3/R2, or in
Google Drive: your application resolves each row to a URL the service can
fetch once, and sends it in batches.

    pip install httpx
    python bulk_index_from_database.py
"""

import os
from collections.abc import Iterable, Iterator

import httpx

BASE_URL = os.environ.get("OVS_BASE_URL", "http://localhost:8000")
COLLECTION = os.environ.get("OVS_COLLECTION", "products")
HEADERS = (
    {"X-API-Key": os.environ["OVS_API_KEY"]} if os.environ.get("OVS_API_KEY") else {}
)

# How many images to send per HTTP call. The server embeds them in batches of
# EMBED_BATCH_SIZE internally, so a few dozen per request is a good balance.
REQUEST_BATCH_SIZE = 50


def fetch_products(offset: int, limit: int) -> list:
    """Replace this with a real query against your database.

    Keep it keyset- or offset-paginated so a backfill of millions of rows does
    not load the whole table into memory.
    """
    rows = [
        {
            "sku": f"SKU-{index:06d}",
            "name": f"Product {index}",
            "category": "shoes" if index % 2 else "bags",
            "bucket": "team-a-products",
            "object_key": f"products/{index}.jpg",
        }
        for index in range(200)
    ]
    return rows[offset : offset + limit]


def signed_url_for(row: dict) -> str:
    """Return a short-lived URL the service can download once.

    For S3/R2/MinIO use your SDK's presigned-URL call; for Google Drive use a
    temporary download link. The service fetches the image, keeps the vector,
    and drops the bytes, so the URL may expire straight afterwards.
    """
    return f"https://{row['bucket']}.example-cdn.com/{row['object_key']}"


def to_index_item(row: dict) -> dict:
    return {
        # Use your own primary key so a search result maps back to a row.
        "id": row["sku"],
        "source": {"type": "url", "value": signed_url_for(row)},
        "metadata": {
            "name": row["name"],
            "category": row["category"],
            # Stable coordinates, never the expiring signed URL, so you can
            # re-sign at render time.
            "storage_provider": "s3",
            "bucket": row["bucket"],
            "object_key": row["object_key"],
        },
    }


def batched(items: Iterable, size: int) -> Iterator:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_all_products() -> Iterator:
    offset = 0
    while True:
        rows = fetch_products(offset, REQUEST_BATCH_SIZE)
        if not rows:
            return
        yield from rows
        offset += len(rows)


def main() -> None:
    indexed_total = 0
    failed_total = 0

    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=300) as client:
        created = client.post("/collections", json={"name": COLLECTION})
        if created.status_code not in (201, 409):
            created.raise_for_status()

        for batch in batched(iter_all_products(), REQUEST_BATCH_SIZE):
            payload = {"images": [to_index_item(row) for row in batch]}
            response = client.post(f"/collections/{COLLECTION}/index", json=payload)
            response.raise_for_status()
            body = response.json()

            indexed_total += body["indexed_count"]
            failed_total += body["failed_count"]

            # Partial failure is normal: a dead URL fails one image, not the batch.
            for error in body["errors"]:
                print(f"  failed {error['id']}: {error['code']} {error['message']}")

            print(f"indexed {indexed_total}, failed {failed_total}")

    print(f"done: {indexed_total} indexed, {failed_total} failed")


if __name__ == "__main__":
    main()
