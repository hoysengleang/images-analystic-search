// End-to-end walkthrough against a local OpenVisionSearch server.
//
//   node quickstart.mjs path/to/query.jpg
//
// Node 18+ only: fetch, FormData, and Blob are built in.

import { readFile } from "node:fs/promises";
import { basename } from "node:path";

const BASE_URL = process.env.OVS_BASE_URL ?? "http://localhost:8000";
const COLLECTION = process.env.OVS_COLLECTION ?? "products";
const AUTH_HEADERS = process.env.OVS_API_KEY
  ? { "X-API-Key": process.env.OVS_API_KEY }
  : {};

const CATALOGUE = [
  {
    id: "shoe_001",
    source: { type: "url", value: "https://picsum.photos/id/21/600/600" },
    display_image_url: "https://picsum.photos/id/21/600/600",
    metadata: { name: "Blue Shoe", category: "shoes", price: 29.99 },
  },
  {
    id: "bag_001",
    source: { type: "url", value: "https://picsum.photos/id/1060/600/600" },
    display_image_url: "https://picsum.photos/id/1060/600/600",
    metadata: { name: "Leather Bag", category: "bags", price: 89.0 },
  },
];

async function call(path, { method = "GET", json, body, allow = [] } = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: {
      ...AUTH_HEADERS,
      ...(json ? { "Content-Type": "application/json" } : {}),
    },
    body: json ? JSON.stringify(json) : body,
  });

  const payload = await response.json();
  if (!response.ok && !allow.includes(response.status)) {
    // Every error shares one shape: { error: { code, message, details } }.
    throw new Error(`${path} -> ${payload?.error?.code}: ${payload?.error?.message}`);
  }
  return payload;
}

const health = await call("/health");
console.log("health:", health);

// 409 just means a previous run already created it.
await call("/collections", { method: "POST", json: { name: COLLECTION }, allow: [409] });

const indexed = await call(`/collections/${COLLECTION}/index`, {
  method: "POST",
  json: { images: CATALOGUE },
});
console.log(`indexed ${indexed.indexed_count}, failed ${indexed.failed_count}`);
for (const error of indexed.errors) {
  console.log(`  ${error.id}: ${error.code} ${error.message}`);
}

const found = await call(`/collections/${COLLECTION}/search`, {
  method: "POST",
  json: { source: CATALOGUE[0].source, top_k: 5, filters: { category: "shoes" } },
});
for (const result of found.results) {
  console.log(`  ${result.score.toFixed(3)}  ${result.id}  ${result.display_image_url}`);
}

const queryImagePath = process.argv[2];
if (queryImagePath) {
  const form = new FormData();
  form.append("image", new Blob([await readFile(queryImagePath)]), basename(queryImagePath));
  form.append("top_k", "5");

  const uploaded = await call(`/collections/${COLLECTION}/search/upload`, {
    method: "POST",
    body: form,
  });
  console.log("upload search:", uploaded.results.map((result) => result.id));
}
