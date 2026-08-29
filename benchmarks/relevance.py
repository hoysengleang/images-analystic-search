"""Measure real search quality and latency with actual model weights.

Builds a catalogue, then queries with degraded copies of each image the way a
shopper's phone photo differs from a catalogue shot: cropped, rotated,
recompressed, and colour-shifted. Reports Recall@K and MRR.

    python benchmarks/relevance.py --catalogue 200 --queries 50

The synthetic catalogue exercises the pipeline honestly but is NOT a substitute
for measuring on a real product catalogue: CLIP behaves differently on natural
photographs. Point --image-dir at real images once you have them.
"""

from __future__ import annotations

import argparse
import colorsys
import random
import statistics
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

SHAPES = ("circle", "square", "triangle", "ring", "bars", "cross")

# The golden angle spreads hues so that consecutive seeds are far apart in
# colour, which keeps every generated product visually distinct.
GOLDEN_ANGLE = 137.508


def _rgb(hue_degrees: float, saturation: float, value: float) -> tuple:
    red, green, blue = colorsys.hsv_to_rgb((hue_degrees % 360) / 360, saturation, value)
    return (int(red * 255), int(green * 255), int(blue * 255))


def synthetic_product(seed: int, size: int = 224) -> Image.Image:
    """A visually unique pseudo-product.

    Every seed gets its own hue, shape, proportions, and accent, so no two
    catalogue entries are near-duplicates. Duplicates would cap Recall@1 at
    the rate of guessing among the twins and make the benchmark meaningless.
    """
    rng = random.Random(seed)
    hue = seed * GOLDEN_ANGLE

    background = _rgb(hue + 180, 0.06 + (seed % 5) * 0.02, 0.97)
    colour = _rgb(hue, 0.62 + (seed % 7) * 0.05, 0.85)
    accent = _rgb(hue + 40 + (seed % 11) * 6, 0.75, 0.7)

    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    shape = SHAPES[seed % len(SHAPES)]
    inset = 24 + (seed % 9) * 4
    box = (inset, inset, size - inset, size - inset)

    if shape == "circle":
        draw.ellipse(box, fill=colour)
    elif shape == "square":
        draw.rectangle(box, fill=colour)
    elif shape == "triangle":
        draw.polygon(
            [(size // 2, inset), (inset, size - inset), (size - inset, size - inset)],
            fill=colour,
        )
    elif shape == "ring":
        draw.ellipse(box, outline=colour, width=12 + (seed % 5) * 4)
    elif shape == "bars":
        bar_count = 3 + seed % 4
        span = (size - 2 * inset) // bar_count
        for index in range(bar_count):
            top = inset + index * span
            draw.rectangle((inset, top, size - inset, top + span // 2), fill=colour)
    else:
        mid, arm = size // 2, 10 + (seed % 5) * 4
        draw.rectangle((mid - arm, inset, mid + arm, size - inset), fill=colour)
        draw.rectangle((inset, mid - arm, size - inset, mid + arm), fill=colour)

    # A per-seed accent stripe in a varying corner adds another axis of identity.
    stripe = 18 + (seed % 4) * 6
    if seed % 4 == 0:
        draw.rectangle((0, size - stripe, size, size), fill=accent)
    elif seed % 4 == 1:
        draw.rectangle((0, 0, size, stripe), fill=accent)
    elif seed % 4 == 2:
        draw.rectangle((0, 0, stripe, size), fill=accent)
    else:
        draw.rectangle((size - stripe, 0, size, size), fill=accent)

    for _ in range(6 + seed % 5):
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        radius = rng.randint(3, 9)
        draw.ellipse((x, y, x + radius, y + radius), fill=accent)

    return image


#: How much a query photo differs from the catalogue shot.
#: "typical" is a decent phone photo; "harsh" stacks every degradation at once
#: and is a worst case, not an expectation.
PROFILES = {
    "typical": {
        "crop": (0.88, 0.98),
        "rotate": 4.0,
        "scale": (0.85, 1.05),
        "brightness": (0.9, 1.1),
        "colour": (0.9, 1.1),
        "blur_chance": 0.25,
        "blur": (0.2, 0.6),
        "quality": (75, 92),
    },
    "harsh": {
        "crop": (0.72, 0.92),
        "rotate": 9.0,
        "scale": (0.6, 1.1),
        "brightness": (0.75, 1.25),
        "colour": (0.75, 1.2),
        "blur_chance": 0.5,
        "blur": (0.3, 1.2),
        "quality": (45, 80),
    },
}


def as_shopper_photo(image: Image.Image, seed: int, profile: str = "typical"):
    """Degrade a catalogue image the way a phone photo would differ."""
    settings = PROFILES[profile]
    rng = random.Random(seed * 977 + 13)
    width, height = image.size

    scale = rng.uniform(*settings["crop"])
    crop_w, crop_h = int(width * scale), int(height * scale)
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    photo = image.crop((left, top, left + crop_w, top + crop_h))

    photo = photo.rotate(
        rng.uniform(-settings["rotate"], settings["rotate"]),
        expand=True,
        fillcolor=(240, 240, 240),
    )
    resize = rng.uniform(*settings["scale"])
    photo = photo.resize((int(photo.width * resize), int(photo.height * resize)))
    photo = ImageEnhance.Brightness(photo).enhance(rng.uniform(*settings["brightness"]))
    photo = ImageEnhance.Color(photo).enhance(rng.uniform(*settings["colour"]))
    if rng.random() < settings["blur_chance"]:
        photo = photo.filter(ImageFilter.GaussianBlur(rng.uniform(*settings["blur"])))

    buffer = BytesIO()
    photo.convert("RGB").save(
        buffer, format="JPEG", quality=rng.randint(*settings["quality"])
    )
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def load_catalogue(image_dir: Path | None, count: int) -> list:
    if image_dir is None:
        return [(f"synthetic_{i:05d}", synthetic_product(i)) for i in range(count)]

    paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[:count]
    return [(path.stem, Image.open(path).convert("RGB")) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=int, default=200)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="typical")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Real product images. Strongly preferred over synthetic.",
    )
    args = parser.parse_args()

    from app.embedding.providers.openclip_provider import OpenCLIPProvider

    print(f"loading {args.model} / {args.pretrained} ...")
    load_start = time.perf_counter()
    provider = OpenCLIPProvider(
        model_name=args.model,
        model_pretrained=args.pretrained,
        vector_size=512,
    )
    provider.warmup()
    print(f"  ready in {time.perf_counter() - load_start:.1f}s")

    catalogue = load_catalogue(args.image_dir, args.catalogue)
    print(f"catalogue: {len(catalogue)} images")

    index_start = time.perf_counter()
    vectors = []
    for start in range(0, len(catalogue), args.batch_size):
        batch = catalogue[start : start + args.batch_size]
        vectors.extend(provider.embed_images([image for _, image in batch]))
    index_seconds = time.perf_counter() - index_start
    ids = [product_id for product_id, _ in catalogue]

    print(
        f"indexing : {index_seconds:.1f}s total, "
        f"{index_seconds / len(catalogue) * 1000:.0f}ms/image "
        f"(batch {args.batch_size})"
    )

    query_seeds = random.Random(7).sample(
        range(len(catalogue)), min(args.queries, len(catalogue))
    )
    ranks, embed_times, search_times = [], [], []

    for position in query_seeds:
        expected_id, source_image = catalogue[position]
        photo = as_shopper_photo(source_image, position, args.profile)

        embed_start = time.perf_counter()
        query_vector = provider.embed_image(photo)
        embed_times.append((time.perf_counter() - embed_start) * 1000)

        search_start = time.perf_counter()
        scored = sorted(
            (
                (sum(a * b for a, b in zip(query_vector, candidate)), candidate_id)
                for candidate_id, candidate in zip(ids, vectors)
            ),
            reverse=True,
        )
        search_times.append((time.perf_counter() - search_start) * 1000)

        rank = next(
            (
                i + 1
                for i, (_, candidate_id) in enumerate(scored)
                if candidate_id == expected_id
            ),
            None,
        )
        ranks.append(rank)

    found = [rank for rank in ranks if rank is not None]

    def recall_at(k: int) -> float:
        return sum(1 for rank in found if rank <= k) / len(ranks)

    print()
    print(
        f"queries  : {len(ranks)} '{args.profile}' photos against "
        f"{len(catalogue)} catalogue images"
    )
    print(f"  Recall@1  : {recall_at(1):.2%}")
    print(f"  Recall@5  : {recall_at(5):.2%}")
    print(f"  Recall@10 : {recall_at(10):.2%}")
    print(f"  MRR       : {sum(1 / r for r in found) / len(ranks):.3f}")
    print(f"  median rank of the correct item: {statistics.median(found):.0f}")
    print()
    print(f"  query embed p50 : {statistics.median(embed_times):.0f}ms")
    print(
        f"  brute search p50: {statistics.median(search_times):.1f}ms "
        f"(pure python, over {len(vectors)} vectors)"
    )


if __name__ == "__main__":
    main()
