# 0007 — Region search is query-side only, and detection ships switched off

**Status:** Accepted · 2026-09-05 · Deviates from Section 13 and Section 25 sequencing

## Context

A shopper photographs a handbag on a cluttered table. Embedding the whole frame
produces a vector describing the table, the mug and the laptop as much as the
bag, and the match is weak. Section 5 accepts this for the first release —
"Guarantee exact SKU recognition from every photograph" is a stated non-goal —
but it is the most common way a real query fails.

The specification already requires half the remedy and does not have it. Line
424 asks for "an optional validated crop rectangle", and lines 505–508 define
`crop_x`, `crop_y`, `crop_width` and `crop_height` as normalized query
parameters. Nothing in `app/` implemented them.

The other half is deferred, twice. Line 426 calls automatic object detection a
Phase 2 feature, and line 845 forbids adding it "before the basic end-to-end
search is reliable" — a sequencing rule, not a rejection. Line 818 puts
index-side "configurable multi-vector product representations" in Milestone 4.

## Decision

Region search is implemented on the **query** side only.

- The crop rectangle is built as specified, on every image-query endpoint.
- An optional detector finds the objects in a query image and searches each one.
  It is registered behind `DETECTOR_PROVIDER`, which defaults to `none`.
- Several query regions are folded into one result list by keeping each image's
  best score across regions.
- Catalogue images continue to produce exactly one vector each. Nothing about
  stored data, point identity or the index changes.

Automatic detection is therefore a deviation from line 426, taken deliberately
and recorded here. The sequencing rule in line 845 is respected in substance:
detection is off unless a deployment turns it on, and a deployment that leaves
it off is byte-for-byte the system that existed before.

The detector is OWL-ViT v1 (`google/owlvit-base-patch32`, Apache 2.0), run
through ONNX Runtime. OWLv2 scores better but its reference preprocessing is a
scikit-image resize that Pillow cannot reproduce, so it could not be verified
for exact parity without depending on scipy. Ultralytics YOLO was rejected
outright: AGPL-3.0 is incompatible with this project's MIT licence and with
self-hosting by commercial users.

## Rationale

Index-side crops are the expensive half and the one the specification defers
furthest, for reasons that hold today:

- They require product grouping, which exists only in the unwired SQLite engine.
  Without it, indexing five crops per catalogue image makes results *worse* —
  five near-identical rows for one photo, with no dedup on the live path.
- Point identity is one Qdrant point per image (`uuid5(collection:image_id)`),
  and five call sites depend on it.
- ADR 0005 budgets 512-dimension vectors at 2 KB each. K crops per image
  multiplies the index by K, against a stated RAM ceiling.

The query side has none of those costs, and it addresses the more common case:
merchant catalogue photographs are usually clean, and the cluttered picture is
the shopper's.

The fold rule was chosen to be the one that already has to exist.
`visual_score(product) = max(similarity(query, product_image_i))` is what line
447 mandates for grouping a product's images; keeping each image's best score
across query regions is the same `max`, and because `max` is associative the two
compose without rework when grouping lands above.

## Consequences

- Detection costs one extra model pass per query — measured at 203 ms on CPU
  for a 900×600 image, against 19 ms to embed one image — plus one embedding and
  one engine round trip per region. A search goes from roughly 25 ms to roughly
  220 ms. An explicit crop costs nothing and is exact, so a client that already
  knows where the object is should send one.
- `MAX_QUERY_REGIONS` defaults to **one**. Folding by best score is monotonic:
  an extra region can raise a product's score but never lower it, so a weak
  region lifts whatever it resembles. Measured on a bag beside a laptop, the
  strongest region alone left the bag 0.257 clear of the runner-up, while a
  second 0.09-confidence region lifted the laptop and cut the lead to 0.073 —
  worse than searching the whole frame. Multiple regions are a "find everything
  in this photo" feature, not a "find this product" one.
- A detector that finds nothing falls back to searching the whole image. Refusing
  to answer would be a worse outcome than a weaker query; the response reports
  the regions actually searched so the difference is visible.
- Detection quality depends heavily on the prompt. Measured on one cluttered
  scene, naming the object scored 0.73 while a generic prompt scored 0.16 and a
  bare noun found nothing. Callers should pass `detect_prompt`.
- Relevance has **not** been measured on real photographs. Section 21 forbids
  accepting a ranking change on hand-picked examples, so `detect` must not be
  recommended as a default until it is measured with `benchmarks/relevance.py`
  on a real catalogue.
- Cluttered *catalogue* images remain unhandled. That is index-side work and
  stays behind product grouping and the `/v1` API.
