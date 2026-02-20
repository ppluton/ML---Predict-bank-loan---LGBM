# Fiche pédagogique — Stack MLOps : Credit Scoring

> **Contexte** : Ce projet prédit le risque de défaut de crédit (Home Credit Default Risk). Un modèle LightGBM, entraîné sur 307 511 clients et 419 features, est déployé en production via FastAPI. Cette fiche explique les concepts clés de la stack et leur rôle concret dans le projet.

---

## Table des matières

1. [Le Data Drift — comprendre la dérive des données](#1-le-data-drift)
2. [Grafana — visualiser les métriques en temps réel](#2-grafana)
3. [FastAPI — servir le modèle en production](#3-fastapi)
4. [LightGBM + MLflow — modéliser et tracer les expériences](#4-lightgbm--mlflow)
5. [Neon PostgreSQL — stocker les prédictions](#5-neon-postgresql)
6. [Streamlit — le dashboard interactif métier](#6-streamlit)
7. [Evidently AI — rapports de drift automatisés](#7-evidently-ai)
8. [Fluentd — agréger les logs Docker](#8-fluentd)
9. [Docker Compose — orchestrer tous les services](#9-docker-compose)
10. [GitHub Actions — automatiser le CI/CD](#10-github-actions)
11. [Vue d'ensemble : le flux de données complet](#11-vue-densemble)

---

## 1. Le Data Drift

### Qu'est-ce que c'est ?

Imaginez que vous apprenez à reconnaître des chats en regardant uniquement des chats européens. Puis on vous présente des chats japonais : ils ressemblent à des chats, mais leurs caractéristiques (taille, fourrure, comportement) ont une distribution légèrement différente. Vous commencez à faire des erreurs sans vous en rendre compte.

C'est exactement le **data drift** : les données qui arrivent en production ont une **distribution statistique différente** de celles qui ont servi à entraîner le modèle.

### Les différents types de drift

```
┌─────────────────────────────────────────────────────────────┐
│                        TYPES DE DRIFT                       │
├─────────────────┬───────────────────────────────────────────┤
│ Covariate drift │ Les features X changent                   │
│ (feature drift) │ → EXT_SOURCE_MEAN diminue en production   │
│                 │ → Le cas le plus courant, celui du projet │
├─────────────────┼───────────────────────────────────────────┤
│ Concept drift   │ La relation X → Y change                  │
│                 │ → Une crise économique change qui fait     │
│                 │   défaut, même avec les mêmes features     │
│                 │ → Impossible à détecter sans labels        │
├─────────────────┼───────────────────────────────────────────┤
│ Sudden drift    │ Changement brutal (ex: COVID, loi)        │
├─────────────────┼───────────────────────────────────────────┤
│ Gradual drift   │ Changement progressif sur des semaines    │
└─────────────────┴───────────────────────────────────────────┘
```

### Le test statistique utilisé : Kolmogorov-Smirnov (KS)

Le test KS compare deux groupes de données en regardant leurs **fonctions de répartition cumulée** (CDF — Cumulative Distribution Function). La CDF répond à la question : "Quelle fraction des données est inférieure à X ?"

```
CDF d'une feature (ex: AMT_ANNUITY)

Proportion
cumulée (%)
   100% │                              ████████
        │                         ████
        │                    █████
    50% │              ██████                      ← Données entraînement
        │         █████                            ← Données production
        │    █████   ↑
     0% └────┴──────────────────────────────────→ Valeur de AMT_ANNUITY
                  KS stat = max écart entre les deux courbes
```

**La statistique KS** = l'écart maximum entre les deux courbes (entre 0 et 1)
- KS ≈ 0 : distributions identiques, pas de drift
- KS ≈ 0.5 : distributions très différentes, drift significatif

**La p-value** : probabilité d'observer cet écart par hasard
- p-value < 0.05 → l'écart n'est pas dû au hasard → **drift détecté**

Avantage du test KS : **non-paramétrique** — il ne suppose aucune forme particulière pour la distribution (pas besoin d'une loi normale).

### Dans ce projet

```
Données de référence               Données de production
(entraînement, 5 000 clients)      (prédictions réelles en DB)
data/training_preprocessed.csv  →  table predictions (Neon Postgres)
                  │                         │
                  └──────────┬──────────────┘
                             ↓
                  monitoring/drift.py
                  compute_drift_report()
                  • KS test sur chaque feature
                  • Résultat : KS stat + p-value par feature
                             ↓
                  scripts/run_drift_detection.py
                  (lancé régulièrement, fenêtre 24h)
                             ↓
                  ┌──────────────────────────┐
                  │  < 10% features driftées │ → OK     (status: ok)
                  │  10–30% driftées         │ → Alerte (status: warning)
                  │  > 30% driftées          │ → Urgent (status: alert)
                  └──────────────────────────┘
                             ↓
                  Stocké dans drift_reports (Postgres)
```

**Features les plus surveillées** (les plus importantes pour le modèle) :
- `EXT_SOURCE_MEAN` — score crédit externe moyen
- `EXT_SOURCE_1` — premier score externe
- `AMT_ANNUITY` — montant de l'annuité
- `DAYS_BIRTH` — âge du client

**Simulation de drift** : le script `scripts/simulate_production_traffic.py` envoie des batches de données avec drift artificiel pour tester le système :
- Batches 1-3 : données propres (pas de drift)
- Batches 4-6 : drift graduel (bruit gaussien croissant)
- Batches 7-8 : drift soudain (décalage brutal des distributions)

---

## 2. Grafana

### Qu'est-ce que c'est ?

Grafana, c'est comme le **tableau de bord d'une voiture** — mais pour votre système informatique. Il affiche des métriques en temps réel sous forme de graphiques, jauges et alertes, en interrogeant directement votre base de données.

```
Base de données         Grafana             Humain
(Neon Postgres)  →  [Requêtes SQL]  →  [Graphiques] → Décisions
                      toutes les 30s
```

**Concepts clés :**
- **Datasource** : la source de données (ici, Neon PostgreSQL)
- **Panel** : un graphique ou indicateur dans le dashboard
- **Dashboard** : ensemble de panels organisés
- **Provisioning** : configuration automatique via des fichiers (pas besoin de cliquer dans l'UI)

### Dans ce projet

```
docker-compose.yml
  grafana:
    image: grafana/grafana:11.0.0
    ports: 3000:3000
    volumes:
      - ./grafana/provisioning  → configure la datasource automatiquement
      - ./grafana/dashboards    → charge mlops-monitoring.json au démarrage
```

**Datasource** : connexion à Neon PostgreSQL (configurée via variables d'environnement `NEON_HOST`, `NEON_USER`, `NEON_PASSWORD`)

**Ce que Grafana interroge :**
```sql
-- Exemple de requête pour la latence API
SELECT created_at, inference_time_ms
FROM predictions
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at;

-- Exemple pour le volume de prédictions
SELECT DATE_TRUNC('hour', created_at) as heure, COUNT(*) as nb
FROM predictions
GROUP BY heure
ORDER BY heure;
```

**Différence avec Streamlit :**

| | Grafana | Streamlit |
|---|---|---|
| **Usage** | Monitoring opérationnel | Dashboard métier interactif |
| **Langage** | Requêtes SQL | Python + ML |
| **Public** | DevOps, infrastructure | Data Scientists, métier |
| **Rafraîchissement** | Automatique (toutes les Ns) | Manuel ou cache 20s |
| **Interactivité** | Filtres temporels | Simulation de drift, scoring client |

Grafana surveille **que le système fonctionne bien**. Streamlit aide à **comprendre ce que le modèle fait**.

---

## 3. FastAPI

### Qu'est-ce que c'est ?

FastAPI est un **framework web Python** pour créer des APIs REST. Une API REST, c'est comme un guichet : vous envoyez une requête (avec des données), le guichet traite et vous répond.

```
Client (navigateur, app, autre service)
        │
        │  POST /predict
        │  {"SK_ID_CURR": 100001, "features": {...}}
        ↓
    FastAPI (api/app.py)
        │
        │  1. Valide les données (Pydantic)
        │  2. Transforme les features (scaler.pkl)
        │  3. Prédit (model.pkl)
        │  4. Enregistre en DB
        │
        ↓
    {"probability": 0.23, "decision": "APPROVED", ...}
```

### Dans ce projet

**4 endpoints** dans [api/app.py](../api/app.py) :

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie que l'API et le modèle sont chargés |
| `/predict` | POST | Reçoit les features, retourne la décision |
| `/model-info` | GET | Métadonnées du modèle (seuil, AUC, coûts) |
| `/internal/logs` | POST | Reçoit les logs de Fluentd (usage interne) |

**Design fail-safe** : si la base de données Neon est indisponible, l'API continue de fonctionner et log localement dans `monitoring/predictions_log.jsonl`. L'API ne tombe jamais à cause d'une erreur de DB.

**Seuil de décision** : 0.494 (et non 0.5) — optimisé pour minimiser le coût métier asymétrique (un défaut manqué coûte 10× plus qu'un bon client refusé).

---

## 4. LightGBM + MLflow

### LightGBM : le moteur de prédiction

LightGBM est un algorithme de **gradient boosting** — il combine des centaines d'arbres de décision simples pour faire une prédiction robuste. Il est particulièrement efficace sur les données tabulaires.

```
Client (419 features)
   ↓
Arbre 1 : EXT_SOURCE_MEAN > 0.5 → probabilité 0.2
   ↓
Arbre 2 : AMT_ANNUITY > 15000 → ajustement +0.05
   ↓
Arbre 3 : DAYS_BIRTH < -12000 → ajustement -0.03
   ↓
   ...  (centaines d'arbres)
   ↓
Probabilité finale de défaut : 0.23
Seuil 0.494 : 0.23 < 0.494 → APPROVED
```

**Performances** : AUC = 0.785, coût métier = 0.49 (vs 0.5 du modèle naïf)

### MLflow : le journal de bord des expériences

MLflow enregistre **tout ce qui se passe pendant l'entraînement** pour permettre la reproductibilité et la comparaison des modèles.

```
notebooks/03_modeling_with_MLFLOW.ipynb
   │
   ├── Test 5 modèles (LogReg, RF, XGBoost, LightGBM...)
   │     → MLflow enregistre : paramètres, métriques, confusion matrix
   │
   ├── GridSearchCV sur LightGBM
   │     → MLflow trace chaque combinaison testée
   │
   └── Meilleur modèle → Model Registry
              ↓
         scripts/export_model.py
              ↓
         artifacts/model.pkl
         artifacts/scaler.pkl
         artifacts/feature_names.json
         artifacts/model_metadata.json
```

**MLflow UI** : accessible localement pendant le développement pour comparer les runs.

---

## 5. Neon PostgreSQL

### Qu'est-ce que c'est ?

Neon est une base de données **PostgreSQL serverless** dans le cloud. "Serverless" signifie que vous ne gérez pas de serveur — la base démarre automatiquement quand vous l'utilisez et s'éteint quand elle est inactive (économie de ressources).

PostgreSQL est une base de données relationnelle — les données sont organisées en tables avec des colonnes et des lignes, interrogeables via SQL.

### Dans ce projet

**4 tables** dans [api/database.py](../api/database.py) :

```
┌─────────────────────────────────────────────────────────────┐
│  predictions                                                │
│  ─────────────────────────────────────────────────────────  │
│  id | sk_id_curr | probability | decision | features(JSONB) │
│     |            |             |          | inference_time  │
│     |            |             |          | created_at      │
│                                                             │
│  Chaque appel à /predict crée une ligne ici                 │
│  Les 419 features sont stockées en JSON brut               │
│  → Permet de rejouer le drift detection plus tard          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  drift_reports                                              │
│  ─────────────────────────────────────────────────────────  │
│  window_start | window_end | n_features_drifted | status   │
│  drift_details (JSONB) : [{feature, ks_stat, p_value}...]  │
│                                                             │
│  Créé par scripts/run_drift_detection.py                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  api_logs                                                   │
│  ─────────────────────────────────────────────────────────  │
│  timestamp | level | event | message | extra (JSONB)        │
│                                                             │
│  Créé par Fluentd (logs système)                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  training_reference                                         │
│  ─────────────────────────────────────────────────────────  │
│  feature_name | mean | std | median | q25 | q75             │
│                                                             │
│  Statistiques pré-calculées des données d'entraînement      │
│  → Référence pour la comparaison de drift                   │
└─────────────────────────────────────────────────────────────┘
```

**JSONB** : format JSON stocké de manière binaire — permet de stocker des structures flexibles (les 419 features varient potentiellement) tout en gardant des requêtes rapides.

---

## 6. Streamlit

### Qu'est-ce que c'est ?

Streamlit transforme un script Python en **application web interactive** sans avoir à écrire du HTML ou du JavaScript. Idéal pour les Data Scientists qui veulent partager leurs analyses.

### Dans ce projet

**5 onglets** dans [monitoring/dashboard.py](../monitoring/dashboard.py) (1300+ lignes) :

```
┌────────────────────────────────────────────────────────┐
│  STREAMLIT DASHBOARD (port 8501)                       │
├──────────┬───────────┬──────────┬───────────┬──────────┤
│ Scoring  │  Scores & │  Perf.   │   Drift   │  Modèle  │
│ client   │ Décisions │  API     │           │          │
├──────────┼───────────┼──────────┼───────────┼──────────┤
│ Saisir   │ Histog.   │ Latence  │ KS bars   │ AUC      │
│ features │ proba     │ P50/P95  │ rouge=KO  │ Seuil    │
│ par ID   │ Taux acc. │ Timeline │ Evidently │ Coûts    │
│ ou manu. │ Timeline  │ (API ✓?) │ HTML      │ Features │
└──────────┴───────────┴──────────┴───────────┴──────────┘
```

**Onglet Drift** en détail :
1. Sélection du type de drift à simuler (gradual, sudden...)
2. Intensité (0 → 1)
3. Calcul KS → bar chart (rouge = drift, vert = OK)
4. Comparaison des distributions (référence vs production) pour les 3 features les plus driftées
5. Bouton "Générer rapport Evidently" → crée le HTML interactif

---

## 7. Evidently AI

### Qu'est-ce que c'est ?

Evidently AI est une bibliothèque Python qui génère des **rapports HTML interactifs** pour le monitoring de modèles ML. Elle encapsule des dizaines de tests statistiques et les présente sous forme visuelle.

### Dans ce projet

```python
# Dans monitoring/dashboard.py et notebooks/04
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=df_reference,   # données entraînement
    current_data=df_production     # prédictions récentes
)
report.save_html("monitoring/drift_report_evidently.html")
```

**DataDriftPreset** inclut automatiquement :
- Résumé global (% features driftées)
- Distribution détaillée par feature (histogrammes superposés)
- Statistique KS, p-value, et verdict drift/pas drift
- Corrélations entre features

**Output** : `monitoring/drift_report_evidently.html` — fichier HTML standalone consultable dans n'importe quel navigateur, sans serveur.

**Complémentarité avec le test KS custom** :
- Le test KS dans `monitoring/drift.py` est utilisé pour la logique métier (seuils, stockage en DB, alertes)
- Evidently génère la visualisation enrichie pour l'analyse humaine

---

## 8. Fluentd

### Qu'est-ce que c'est ?

Fluentd est un **agrégateur de logs**. Un log, c'est un message enregistré par une application ("j'ai reçu une requête", "j'ai fait une prédiction", "une erreur s'est produite"). Fluentd collecte ces messages de différentes sources et les envoie vers une destination.

### Dans ce projet

```
API FastAPI (api/app.py)
   │
   │  Écrit des logs JSON sur stdout :
   │  {"timestamp": "...", "event": "prediction", "inference_time_ms": 45}
   │
   ↓ (Docker log driver)
Docker Engine
   │  Capture tous les stdout/stderr des conteneurs
   │
   ↓ (driver: fluentd, port 24224)
Fluentd (conteneur séparé, fluentd/fluent.conf)
   │
   │  1. Reçoit les logs bruts
   │  2. Parse le JSON
   │  3. Bufferise 5 secondes
   │
   ↓ (HTTP POST toutes les 5 secondes)
POST api:8000/internal/logs
   │
   ↓
Neon Postgres → table api_logs
```

**Pourquoi ce design ?** L'API n'écrit pas directement dans Postgres depuis son stdout — c'est Fluentd qui fait le pont. Cela découple la collecte de logs de la logique métier et permet de changer la destination des logs sans toucher au code de l'API.

**Résilience** : si Fluentd est indisponible, les logs restent sur stdout (Docker les garde). Si Postgres est indisponible, Fluentd bufferise et réessaie (3 tentatives).

---

## 9. Docker Compose

### Qu'est-ce que c'est ?

Docker permet d'empaqueter une application et toutes ses dépendances dans un **conteneur** — une boîte isolée qui fonctionne de manière identique sur n'importe quelle machine.

Docker Compose orchestre **plusieurs conteneurs** qui fonctionnent ensemble, en définissant leurs interactions dans un seul fichier YAML.

### Dans ce projet

```
docker-compose.yml → 4 services

┌──────────────────────────────────────────────────────────┐
│                   docker-compose up                      │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐                    │
│  │  API        │    │  Dashboard   │                    │
│  │  (port 8000)│    │  (port 8501) │                    │
│  │  FastAPI +  │    │  Streamlit   │                    │
│  │  LightGBM   │    │  5 onglets   │                    │
│  └──────┬──────┘    └──────┬───────┘                    │
│         │                  │                            │
│  ┌──────▼──────┐    ┌──────▼───────┐                    │
│  │  Fluentd    │    │  Grafana     │                    │
│  │  (port 24224│    │  (port 3000) │                    │
│  │  Log aggr.  │    │  Dashboards  │                    │
│  └─────────────┘    └──────────────┘                    │
│                                                          │
│  Les 4 services partagent le même réseau Docker          │
│  L'API et le Dashboard partagent predictions_log.jsonl   │
└──────────────────────────────────────────────────────────┘
```

**Images Docker** :
- `Dockerfile` → image API (FastAPI + model.pkl)
- `Dockerfile.dashboard` → image Dashboard (Streamlit)
- `grafana/grafana:11.0.0` → image officielle Grafana
- Image Fluentd custom (avec plugin pour HTTP output)

**Dépendances** :
- Dashboard `depends_on: [api]` → l'API démarre avant le dashboard
- API `depends_on: [fluentd]` → Fluentd est prêt avant que l'API envoie des logs

---

## 10. GitHub Actions

### Qu'est-ce que c'est ?

GitHub Actions est un service CI/CD (Continuous Integration / Continuous Deployment). À chaque `git push`, il exécute automatiquement une série de tâches : tests, build, déploiement.

**CI** = on vérifie que le code ne casse rien
**CD** = on déploie automatiquement si tout passe

### Dans ce projet

**Fichier** : [.github/workflows/ci-cd.yml](../.github/workflows/ci-cd.yml)

```
git push → GitHub Actions
│
├── Stage 1 : TEST
│   ├── ruff check . → vérification du style Python (lint)
│   └── pytest tests/ → 19 tests automatisés
│       ├── test_api.py     (6 tests : /health, /predict, /model-info)
│       ├── test_predict.py (inférence du modèle)
│       └── test_drift.py   (8 tests : KS test, simulation)
│
├── Stage 2 : BUILD (si tests OK)
│   ├── docker build -t credit-scoring .
│   └── Health check : curl /health avec 5 essais (retry loop)
│
└── Stage 3 : DEPLOY (si build OK, sur branche main uniquement)
    └── POST Render.com API → déclenche un déploiement cloud
```

**Render.com** : plateforme cloud qui héberge l'API et le Dashboard en production.

---

## 11. Vue d'ensemble : le flux de données complet

```
                    ┌──────────────────────────────┐
                    │  ENTRAÎNEMENT (offline)       │
                    │  notebooks/03_modeling...     │
                    │  LightGBM + MLflow            │
                    │  → artifacts/model.pkl        │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  API FASTAPI (port 8000)      │
                    │  POST /predict                │
                    │  {SK_ID_CURR, features}       │
                    └──┬───────────┬───────────┬───┘
                       │           │           │
              ┌────────▼──┐ ┌─────▼────┐ ┌───▼────────┐
              │ JSONL     │ │ Neon DB  │ │ Réponse    │
              │ (backup   │ │predictions│ │ au client  │
              │  local)   │ │ table    │ │            │
              └───────────┘ └────┬─────┘ └────────────┘
                                 │
                  ┌──────────────┼──────────────────────┐
                  │              │                      │
         ┌────────▼────┐ ┌──────▼──────┐ ┌────────────▼────┐
         │  Fluentd    │ │  Drift      │ │  Grafana        │
         │  api_logs   │ │  Detection  │ │  (port 3000)    │
         │  table      │ │  (KS test)  │ │  Dashboards SQL │
         └─────────────┘ └──────┬──────┘ └─────────────────┘
                                │
                       ┌────────▼──────────┐
                       │  Streamlit        │
                       │  (port 8501)      │
                       │  5 onglets        │
                       │  + Evidently AI   │
                       └───────────────────┘
```

**Résumé des rôles :**

| Composant | Rôle | Port/Fichier |
|-----------|------|-------------|
| **LightGBM** | Modèle de prédiction de crédit | `artifacts/model.pkl` |
| **MLflow** | Tracking des expériences ML | `notebooks/mlruns/` |
| **FastAPI** | Serveur de prédiction (API REST) | `8000` |
| **Neon Postgres** | Base de données cloud (prédictions, logs, drift) | Cloud |
| **Fluentd** | Agrégateur de logs Docker → Postgres | `24224` |
| **Streamlit** | Dashboard interactif (scoring, drift, perf.) | `8501` |
| **Evidently AI** | Rapports HTML de data drift | `.html` |
| **Grafana** | Monitoring temps réel (métriques SQL) | `3000` |
| **Docker Compose** | Orchestration des 4 services | `docker-compose.yml` |
| **GitHub Actions** | CI/CD automatique (test → build → deploy) | `.github/workflows/` |
| **Render.com** | Hébergement cloud en production | Cloud |
