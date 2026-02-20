-- Schema de la base de données de monitoring MLOps
-- Exécuté via scripts/init_db.py

-- Prédictions : inputs (features JSONB), outputs, temps d'exécution
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    sk_id_curr      INTEGER NOT NULL,
    probability     DOUBLE PRECISION NOT NULL,
    prediction      INTEGER NOT NULL,
    decision        VARCHAR(10) NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    inference_time_ms DOUBLE PRECISION NOT NULL,
    features        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_sk_id ON predictions (sk_id_curr);

-- Rapports de drift par fenêtre temporelle
CREATE TABLE IF NOT EXISTS drift_reports (
    id                  SERIAL PRIMARY KEY,
    report_date         TIMESTAMPTZ DEFAULT NOW(),
    window_start        TIMESTAMPTZ NOT NULL,
    window_end          TIMESTAMPTZ NOT NULL,
    n_predictions       INTEGER NOT NULL,
    n_features_analyzed INTEGER NOT NULL DEFAULT 0,
    n_features_drifted  INTEGER NOT NULL,
    drift_percentage    DOUBLE PRECISION NOT NULL,
    drift_details       JSONB NOT NULL,
    status              VARCHAR(20) DEFAULT 'ok'
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_date ON drift_reports (report_date);

-- Logs API (alimenté par Fluentd)
CREATE TABLE IF NOT EXISTS api_logs (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    level       VARCHAR(10) DEFAULT 'INFO',
    event       VARCHAR(50),
    message     TEXT,
    extra       JSONB
);

CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp ON api_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_api_logs_event ON api_logs (event);

-- Statistiques de référence du training data (calculées une fois)
CREATE TABLE IF NOT EXISTS training_reference (
    id           SERIAL PRIMARY KEY,
    feature_name VARCHAR(200) UNIQUE NOT NULL,
    mean         DOUBLE PRECISION,
    std          DOUBLE PRECISION,
    median       DOUBLE PRECISION,
    q25          DOUBLE PRECISION,
    q75          DOUBLE PRECISION,
    n_samples    INTEGER,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
