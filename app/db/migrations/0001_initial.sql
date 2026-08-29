-- Milestone 0 schema.
--
-- Every tenant-owned table carries tenant_id explicitly, even where it could
-- be derived through a join. Repositories filter on it directly so a missing
-- join can never widen a query across tenants.

CREATE TABLE tenants (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL
);

CREATE TABLE api_keys (
    id            TEXT PRIMARY KEY,
    -- NULL tenant_id means an administrative key, which owns no tenant data.
    tenant_id     TEXT REFERENCES tenants (id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    key_hash      TEXT NOT NULL UNIQUE,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT,
    CHECK (
        (is_admin = 1 AND tenant_id IS NULL)
        OR (is_admin = 0 AND tenant_id IS NOT NULL)
    )
);

CREATE TABLE sources (
    id              TEXT NOT NULL,
    tenant_id       TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    type            TEXT NOT NULL,
    name            TEXT NOT NULL,
    -- JSON. Secrets belong in a separate store, never here.
    config          TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'ready',
    sync_cursor     TEXT,
    last_synced_at  TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, name)
);

CREATE TABLE products (
    id            TEXT NOT NULL,
    tenant_id     TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    source_id     TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    title         TEXT,
    description   TEXT,
    category      TEXT,
    brand         TEXT,
    price         REAL,
    currency      TEXT,
    in_stock      INTEGER,
    attributes    TEXT NOT NULL DEFAULT '{}',
    source_url    TEXT,
    content_hash  TEXT NOT NULL,
    deleted_at    TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, source_id)
        REFERENCES sources (tenant_id, id) ON DELETE CASCADE,
    UNIQUE (tenant_id, source_id, external_id)
);

CREATE TABLE product_images (
    id                TEXT NOT NULL,
    tenant_id         TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    product_id        TEXT NOT NULL,
    source_uri        TEXT NOT NULL,
    position          INTEGER NOT NULL DEFAULT 0,
    content_hash      TEXT,
    width             INTEGER,
    height            INTEGER,
    media_type        TEXT,
    embedding_status  TEXT NOT NULL DEFAULT 'pending',
    model_version     TEXT,
    error             TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, product_id)
        REFERENCES products (tenant_id, id) ON DELETE CASCADE,
    UNIQUE (tenant_id, id, product_id),
    UNIQUE (tenant_id, product_id, source_uri)
);

CREATE TABLE vectors (
    id             TEXT NOT NULL,
    tenant_id      TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    product_id     TEXT NOT NULL,
    image_id       TEXT NOT NULL,
    -- "provider:model@revision". Vectors from different versions never mix.
    model_version  TEXT NOT NULL,
    dimension      INTEGER NOT NULL,
    normalized     INTEGER NOT NULL DEFAULT 1,
    vector         BLOB NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, image_id, product_id)
        REFERENCES product_images (tenant_id, id, product_id) ON DELETE CASCADE
);

CREATE TABLE jobs (
    id           TEXT NOT NULL,
    tenant_id    TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    source_id    TEXT,
    type         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',
    progress     TEXT NOT NULL DEFAULT '{}',
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, source_id)
        REFERENCES sources (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE search_events (
    id             TEXT NOT NULL,
    tenant_id      TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    search_type    TEXT NOT NULL,
    filters        TEXT NOT NULL DEFAULT '{}',
    result_count   INTEGER NOT NULL DEFAULT 0,
    latency_ms     INTEGER,
    engine         TEXT,
    model_version  TEXT,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id)
);

CREATE TABLE feedback_events (
    id          TEXT NOT NULL,
    tenant_id   TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    search_id   TEXT NOT NULL,
    event       TEXT NOT NULL,
    product_id  TEXT,
    position    INTEGER,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, search_id)
        REFERENCES search_events (tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, product_id)
        REFERENCES products (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX idx_sources_tenant ON sources (tenant_id);
CREATE INDEX idx_products_tenant ON products (tenant_id, deleted_at);
CREATE INDEX idx_products_source ON products (tenant_id, source_id);
CREATE INDEX idx_products_category ON products (tenant_id, category);
CREATE INDEX idx_product_images_product ON product_images (tenant_id, product_id);
CREATE INDEX idx_product_images_status ON product_images (tenant_id, embedding_status);
CREATE INDEX idx_vectors_tenant_model ON vectors (tenant_id, model_version);
CREATE INDEX idx_vectors_product ON vectors (tenant_id, product_id);
CREATE INDEX idx_vectors_image ON vectors (tenant_id, image_id);
CREATE INDEX idx_jobs_tenant ON jobs (tenant_id, status);
CREATE INDEX idx_search_events_tenant ON search_events (tenant_id, created_at);
CREATE INDEX idx_feedback_events_search ON feedback_events (tenant_id, search_id);
