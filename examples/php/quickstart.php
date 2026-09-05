<?php
/**
 * End-to-end walkthrough against a local OpenVisionSearch server.
 *
 *   php quickstart.php path/to/query.jpg
 */

declare(strict_types=1);

$baseUrl    = getenv('OVS_BASE_URL') ?: 'http://localhost:8000';
$collection = getenv('OVS_COLLECTION') ?: 'products';
$apiKey     = getenv('OVS_API_KEY') ?: null;

$catalogue = [
    [
        'id' => 'shoe_001',
        'source' => ['type' => 'url', 'value' => 'https://picsum.photos/id/21/600/600'],
        'display_image_url' => 'https://picsum.photos/id/21/600/600',
        'metadata' => ['name' => 'Blue Shoe', 'category' => 'shoes', 'price' => 29.99],
    ],
    [
        'id' => 'bag_001',
        'source' => ['type' => 'url', 'value' => 'https://picsum.photos/id/1060/600/600'],
        'display_image_url' => 'https://picsum.photos/id/1060/600/600',
        'metadata' => ['name' => 'Leather Bag', 'category' => 'bags', 'price' => 89.0],
    ],
];

/**
 * @param array<string,mixed>|null $json
 * @param array<string,mixed>|null $multipart
 * @param list<int>                $allowStatuses
 * @return array<string,mixed>
 */
function call(
    string $method,
    string $path,
    ?array $json = null,
    ?array $multipart = null,
    array $allowStatuses = []
): array {
    global $baseUrl, $apiKey;

    $headers = [];
    if ($apiKey !== null) {
        $headers[] = 'X-API-Key: ' . $apiKey;
    }

    $handle = curl_init($baseUrl . $path);
    curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($handle, CURLOPT_CUSTOMREQUEST, $method);
    curl_setopt($handle, CURLOPT_TIMEOUT, 300);

    if ($json !== null) {
        $headers[] = 'Content-Type: application/json';
        curl_setopt($handle, CURLOPT_POSTFIELDS, json_encode($json, JSON_THROW_ON_ERROR));
    } elseif ($multipart !== null) {
        // Let curl set the multipart boundary itself.
        curl_setopt($handle, CURLOPT_POSTFIELDS, $multipart);
    }

    curl_setopt($handle, CURLOPT_HTTPHEADER, $headers);

    $body   = curl_exec($handle);
    $status = curl_getinfo($handle, CURLINFO_HTTP_CODE);
    curl_close($handle);

    if ($body === false) {
        throw new RuntimeException("Request to {$path} failed");
    }

    $decoded = json_decode($body, true, 512, JSON_THROW_ON_ERROR);

    if ($status >= 400 && !in_array($status, $allowStatuses, true)) {
        // Every error shares one shape: { error: { code, message, details } }.
        $code    = $decoded['error']['code'] ?? 'UNKNOWN';
        $message = $decoded['error']['message'] ?? 'unknown error';
        throw new RuntimeException("{$path} -> {$code}: {$message}");
    }

    return $decoded;
}

echo "health: " . json_encode(call('GET', '/health')) . PHP_EOL;

// 409 just means a previous run already created it.
call('POST', '/collections', ['name' => $collection], null, [409]);

$indexed = call('POST', "/collections/{$collection}/index", ['images' => $catalogue]);
printf("indexed %d, failed %d%s", $indexed['indexed_count'], $indexed['failed_count'], PHP_EOL);
foreach ($indexed['errors'] as $error) {
    printf("  %s: %s %s%s", $error['id'], $error['code'], $error['message'], PHP_EOL);
}

$found = call('POST', "/collections/{$collection}/search", [
    'source'  => $catalogue[0]['source'],
    'top_k'   => 5,
    'filters' => ['category' => 'shoes'],
]);
foreach ($found['results'] as $result) {
    printf("  %.3f  %s  %s%s", $result['score'], $result['id'], $result['display_image_url'] ?? '', PHP_EOL);
}

$queryImagePath = $argv[1] ?? null;
if ($queryImagePath !== null && is_file($queryImagePath)) {
    $uploaded = call('POST', "/collections/{$collection}/search/upload", null, [
        'image' => new CURLFile($queryImagePath),
        'top_k' => '5',
    ]);
    echo 'upload search: ' . implode(', ', array_column($uploaded['results'], 'id')) . PHP_EOL;
}
