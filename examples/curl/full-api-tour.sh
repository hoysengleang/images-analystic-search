#!/usr/bin/env bash
# Every endpoint, in the order you would actually use them.
#
#   ./full-api-tour.sh [query-image.jpg]
#
# Set OVS_API_KEY if the server runs with API_KEY configured.
set -euo pipefail

BASE_URL="${OVS_BASE_URL:-http://localhost:8000}"
COLLECTION="${OVS_COLLECTION:-tour}"
QUERY_IMAGE="${1:-}"

AUTH=()
[[ -n "${OVS_API_KEY:-}" ]] && AUTH=(-H "X-API-Key: ${OVS_API_KEY}")

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
call() { curl -sS "${AUTH[@]}" "$@"; echo; }

say "1. Is the service up, and which backend is serving?"
call "${BASE_URL}/health?include_qdrant=true"

say "2. Which embedding models can this server use?"
# supports_text tells you whether /search/text will work on a collection
# built with that model.
call "${BASE_URL}/models"

say "3. Create a collection"
# Omit "model" to accept the server defaults. The model, framing, and view
# count are pinned here for the collection's whole life, because vectors built
# differently cannot be compared.
call -X POST "${BASE_URL}/collections" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"${COLLECTION}\"}" || true

say "4. Index images by URL"
# Use your own primary keys as ids: a search result then maps straight back to
# a row in your database. Re-sending the same id overwrites it.
call -X POST "${BASE_URL}/collections/${COLLECTION}/index" \
  -H 'Content-Type: application/json' \
  -d '{
    "images": [
      {
        "id": "shoe_001",
        "source": { "type": "url", "value": "https://picsum.photos/id/21/600/600" },
        "display_image_url": "https://picsum.photos/id/21/600/600",
        "metadata": { "name": "Blue Shoe", "category": "shoes", "price": 29.99, "in_stock": true }
      },
      {
        "id": "bag_001",
        "source": { "type": "url", "value": "https://picsum.photos/id/1060/600/600" },
        "display_image_url": "https://picsum.photos/id/1060/600/600",
        "metadata": { "name": "Leather Bag", "category": "bags", "price": 89.00, "in_stock": false }
      }
    ]
  }'
# Check indexed_count against failed_count: partial success is normal, and
# every failure is reported per image in "errors".

say "5. Index a single uploaded file"
if [[ -n "${QUERY_IMAGE}" && -f "${QUERY_IMAGE}" ]]; then
  call -X POST "${BASE_URL}/collections/${COLLECTION}/index/upload" \
    -F 'id=uploaded_001' \
    -F "image=@${QUERY_IMAGE}" \
    -F 'metadata={"name":"Uploaded item","category":"shoes"}'
else
  echo "   skipped: pass an image path as the first argument"
fi

say "6. Index a whole folder mounted into the container"
# Paths resolve inside the container and must sit under ALLOWED_IMAGE_ROOT.
# Ids come from the path relative to the folder.
call -X POST "${BASE_URL}/collections/${COLLECTION}/index/folder" \
  -H 'Content-Type: application/json' \
  -d '{ "folder_path": "/data/images", "recursive": true }' || true

say "7. Search by image URL, filtered"
# Filters accept a scalar, a list (match any), or a range object.
call -X POST "${BASE_URL}/collections/${COLLECTION}/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "source": { "type": "url", "value": "https://picsum.photos/id/21/600/600" },
    "top_k": 5,
    "filters": { "category": ["shoes", "bags"], "price": { "lte": 100 } }
  }'

say "8. Search by uploaded photo (the scan-to-search path)"
if [[ -n "${QUERY_IMAGE}" && -f "${QUERY_IMAGE}" ]]; then
  call -X POST "${BASE_URL}/collections/${COLLECTION}/search/upload" \
    -F "image=@${QUERY_IMAGE}" -F 'top_k=5'
else
  echo "   skipped: pass an image path as the first argument"
fi

say "9. Search by text, over the same index"
# Text and image scores are on different scales: a strong text match is around
# 0.2-0.35, a strong image match is 0.8+. Do not reuse one min_score for both.
call -X POST "${BASE_URL}/collections/${COLLECTION}/search/text" \
  -H 'Content-Type: application/json' \
  -d '{ "query": "a blue running shoe", "top_k": 5 }'

say "10. Hybrid: this image, but described differently"
call -X POST "${BASE_URL}/collections/${COLLECTION}/search/hybrid" \
  -H 'Content-Type: application/json' \
  -d '{
    "source": { "type": "url", "value": "https://picsum.photos/id/21/600/600" },
    "query": "in brown leather",
    "text_weight": 0.3,
    "top_k": 5
  }'

say "11. Batch search with several photos of the same object"
# "average" fuses them into one stronger query; "separate" returns a result
# set per photo, tagged with its source_index.
call -X POST "${BASE_URL}/collections/${COLLECTION}/search/batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "sources": [
      { "type": "url", "value": "https://picsum.photos/id/21/600/600" },
      { "type": "url", "value": "https://picsum.photos/id/1060/600/600" }
    ],
    "mode": "average",
    "top_k": 5
  }'

say "12. Page through everything indexed"
# Pass the returned next_cursor back as ?cursor= until it is null.
call "${BASE_URL}/collections/${COLLECTION}/images?limit=2"

say "13. Read one record"
call "${BASE_URL}/collections/${COLLECTION}/images/shoe_001"

say "14. Collection statistics"
call "${BASE_URL}/collections/${COLLECTION}/stats"

say "15. List collections"
call "${BASE_URL}/collections"

say "16. Delete one image, then the collection"
call -X DELETE "${BASE_URL}/collections/${COLLECTION}/images/shoe_001"
call -X DELETE "${BASE_URL}/collections/${COLLECTION}"

printf '\nTour complete. Every endpoint above is documented in docs/api-reference.md\n'
