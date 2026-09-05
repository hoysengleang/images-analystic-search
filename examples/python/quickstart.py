"""End-to-end walkthrough against a local OpenVisionSearch server.

pip install httpx
python quickstart.py path/to/query.jpg
"""

import os
import sys
from typing import Optional

import httpx

BASE_URL = os.environ.get("OVS_BASE_URL", "http://localhost:8000")
COLLECTION = os.environ.get("OVS_COLLECTION", "products")
HEADERS = (
    {"X-API-Key": os.environ["OVS_API_KEY"]} if os.environ.get("OVS_API_KEY") else {}
)

CATALOGUE = [
    {
        "id": "shoe_001",
        "source": {"type": "url", "value": "https://picsum.photos/id/21/600/600"},
        "display_image_url": "https://picsum.photos/id/21/600/600",
        "metadata": {"name": "Blue Shoe", "category": "shoes", "price": 29.99},
    },
    {
        "id": "bag_001",
        "source": {"type": "url", "value": "https://picsum.photos/id/1060/600/600"},
        "display_image_url": "https://picsum.photos/id/1060/600/600",
        "metadata": {"name": "Leather Bag", "category": "bags", "price": 89.00},
    },
]


def main(query_image_path: Optional[str]) -> None:
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=120) as client:
        print("health:", client.get("/health").json())

        # 409 just means a previous run already created it.
        created = client.post("/collections", json={"name": COLLECTION})
        if created.status_code not in (201, 409):
            created.raise_for_status()

        indexed = client.post(
            f"/collections/{COLLECTION}/index", json={"images": CATALOGUE}
        ).json()
        print(f"indexed {indexed['indexed_count']}, failed {indexed['failed_count']}")
        for error in indexed["errors"]:
            print(f"  {error['id']}: {error['code']} {error['message']}")

        results = client.post(
            f"/collections/{COLLECTION}/search",
            json={
                "source": CATALOGUE[0]["source"],
                "top_k": 5,
                "filters": {"category": "shoes"},
            },
        ).json()
        for result in results["results"]:
            print(f"  {result['score']:.3f}  {result['id']}  {result['metadata']}")

        # Text search works on the images already indexed above, because
        # OpenCLIP puts text and images in the same vector space.
        by_text = client.post(
            f"/collections/{COLLECTION}/search/text",
            json={"query": "a leather bag", "top_k": 5},
        ).json()
        print("text search:", [r["id"] for r in by_text["results"]])

        # An image query nudged by words: "this, but more like a bag".
        hybrid = client.post(
            f"/collections/{COLLECTION}/search/hybrid",
            json={
                "source": CATALOGUE[0]["source"],
                "query": "a leather bag",
                "text_weight": 0.4,
                "top_k": 5,
            },
        ).json()
        print("hybrid search:", [r["id"] for r in hybrid["results"]])

        if query_image_path:
            with open(query_image_path, "rb") as image_file:
                uploaded = client.post(
                    f"/collections/{COLLECTION}/search/upload",
                    files={"image": (query_image_path, image_file, "image/jpeg")},
                    data={"top_k": "5"},
                ).json()
            print("upload search:", [r["id"] for r in uploaded["results"]])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
