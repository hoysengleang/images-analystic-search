#!/usr/bin/env bash
# End-to-end walkthrough against a local OpenVisionSearch server.
#
#   ./quickstart.sh path/to/query.jpg
set -euo pipefail

BASE_URL="${OVS_BASE_URL:-http://localhost:8000}"
COLLECTION="${OVS_COLLECTION:-products}"
QUERY_IMAGE="${1:-query.jpg}"

# Only sent when the server was started with API_KEY set.
AUTH=()
if [[ -n "${OVS_API_KEY:-}" ]]; then
  AUTH=(-H "X-API-Key: ${OVS_API_KEY}")
fi

echo "==> Health"
curl -sS "${BASE_URL}/health?include_qdrant=true"; echo

echo "==> Create collection '${COLLECTION}' (409 means it already exists)"
curl -sS -X POST "${BASE_URL}/collections" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"${COLLECTION}\"}"; echo

echo "==> Index images by URL"
curl -sS -X POST "${BASE_URL}/collections/${COLLECTION}/index" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "images": [
      {
        "id": "shoe_001",
        "source": { "type": "url", "value": "https://picsum.photos/id/21/600/600" },
        "display_image_url": "https://picsum.photos/id/21/600/600",
        "metadata": { "name": "Blue Shoe", "category": "shoes", "price": 29.99 }
      },
      {
        "id": "bag_001",
        "source": { "type": "url", "value": "https://picsum.photos/id/1060/600/600" },
        "display_image_url": "https://picsum.photos/id/1060/600/600",
        "metadata": { "name": "Leather Bag", "category": "bags", "price": 89.00 }
      }
    ]
  }'; echo

echo "==> Search by URL"
curl -sS -X POST "${BASE_URL}/collections/${COLLECTION}/search" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "source": { "type": "url", "value": "https://picsum.photos/id/21/600/600" },
    "top_k": 5,
    "min_score": 0.5
  }'; echo

echo "==> Search by text (same index, no re-indexing)"
curl -sS -X POST "${BASE_URL}/collections/${COLLECTION}/search/text" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{ "query": "a leather bag", "top_k": 5 }'; echo

echo "==> Hybrid: this image, but described differently"
curl -sS -X POST "${BASE_URL}/collections/${COLLECTION}/search/hybrid" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{
    "source": { "type": "url", "value": "https://picsum.photos/id/21/600/600" },
    "query": "a leather bag",
    "text_weight": 0.4,
    "top_k": 5
  }'; echo

if [[ -f "${QUERY_IMAGE}" ]]; then
  echo "==> Search by upload (${QUERY_IMAGE})"
  curl -sS -X POST "${BASE_URL}/collections/${COLLECTION}/search/upload" "${AUTH[@]}" \
    -F "image=@${QUERY_IMAGE}" \
    -F 'top_k=5'; echo
else
  echo "==> Skipping upload search; no file at ${QUERY_IMAGE}"
fi

echo "==> Collection stats"
curl -sS "${BASE_URL}/collections/${COLLECTION}/stats" "${AUTH[@]}"; echo
