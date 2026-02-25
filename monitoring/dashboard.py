"""Dashboard Streamlit de monitoring - Credit Scoring Platform."""

import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Ajouter la racine du projet au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st

from evidently import Report
from evidently.presets import DataDriftPreset

from monitoring.drift import compute_drift_report, simulate_drift


def _start_api_keepalive():
    """Pinge l'API toutes les 10 minutes pour éviter le cold start (Render free tier)."""
    api_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

    def _ping():
        while True:
            time.sleep(600)  # 10 minutes
            try:
                req = Request(url=f"{api_url}/health", method="GET")
                with urlopen(req, timeout=5) as resp:  # nosec: B310
                    resp.read()
            except Exception:
                pass

    t = threading.Thread(target=_ping, daemon=True)
    t.start()


@st.cache_resource
def _keepalive_started():
    """Démarre le keep-alive une seule fois par instance Streamlit."""
    _start_api_keepalive()
    return True


_keepalive_started()


def load_predictions_from_db() -> "pd.DataFrame | None":
    """Lit les prédictions depuis PostgreSQL si DATABASE_URL est défini."""
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    try:
        conn = psycopg2.connect(db_url)
        df = pd.read_sql(
            """SELECT sk_id_curr AS "SK_ID_CURR",
                      probability,
                      prediction,
                      decision,
                      inference_time_ms,
                      created_at AS "timestamp"
               FROM predictions
               ORDER BY created_at DESC
               LIMIT 10000""",
            conn,
        )
        conn.close()
        return df if len(df) > 0 else None
    except Exception:
        return None


# --- Configuration ---
st.set_page_config(
    page_title="HC Credit Risk | Monitoring",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded",
)
try:
    # Force l'affichage du menu/toolbar Streamlit si une config locale l'avait masque.
    st.set_option("client.toolbarMode", "auto")
except Exception:
    pass

ARTIFACTS_DIR = Path("artifacts")
PREDICTIONS_LOG = Path("monitoring/predictions_log.jsonl")
REFERENCE_DATA = Path("data/test_preprocessed.csv")

# --- Theme premium bank ---
COLORS = {
    "primary": "#1B2A4A",
    "secondary": "#2C4A7C",
    "accent": "#C9A96E",
    "text": "#1F2B44",
    "muted": "#3D4A5C",  # Darkened for better contrast (was #51607A)
    "chart_bg": "#FFFFFF",
    "success": "#1D6A4B",
    "danger": "#8B2D2D",
    "approved": "#1D6A4B",
    "refused": "#8B2D2D",
    "chart_ref": "#1D6A4B",
    "chart_prod": "#8B2D2D",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Base font settings for better readability */
    .stApp {{
        background-color: #F4F1EC;
        font-family: 'Inter', sans-serif;
        color: {COLORS["text"]};
        font-size: 15px;
        line-height: 1.6;
    }}

    /* Ensure minimum font size for all text */
    .stApp * {{
        font-size: 14px !important;
    }}

    .stApp p, .stApp li, .stApp span {{
        font-size: 15px !important;
        line-height: 1.6 !important;
    }}

    [data-testid="stAppViewContainer"] {{
        --text-color: {COLORS["text"]};
        --secondary-text-color: {COLORS["muted"]};
        color: {COLORS["text"]};
    }}

    header[data-testid="stHeader"] {{
        background: {COLORS["primary"]};
    }}

    div[data-testid="stToolbar"] {{
        visibility: visible !important;
        opacity: 1 !important;
    }}

    .main-header {{
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%);
        padding: 2rem 2.5rem;
        border-radius: 0 0 16px 16px;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
    }}

    .main-header h1 {{
        color: white !important;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }}

    .main-header p {{
        color: {COLORS["accent"]};
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }}

    .section-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {COLORS["primary"]};
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS["accent"]};
    }}

    .status-badge {{
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}

    .status-ok {{
        background: rgba(29,106,75,0.15);
        color: {COLORS["success"]};
    }}

    .status-alert {{
        background: rgba(139,45,45,0.15);
        color: {COLORS["danger"]};
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(27,42,74,0.08);
        border: 1px solid rgba(27,42,74,0.06);
    }}

    div[data-testid="stMetric"] label {{
        color: {COLORS["muted"]} !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500 !important;
    }}

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {COLORS["primary"]} !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }}

    div[data-testid="stWidgetLabel"] > label,
    div[data-testid="stWidgetLabel"] > div,
    div[data-testid="stWidgetLabel"] p {{
        color: {COLORS["text"]} !important;
        font-weight: 500;
        font-size: 14px !important;
    }}

    div[role="radiogroup"] label,
    div[role="radiogroup"] span,
    div[data-baseweb="radio"] label,
    div[data-baseweb="radio"] span,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p {{
        color: {COLORS["text"]} !important;
        font-size: 14px !important;
    }}

    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div {{
        background-color: #FFFFFF !important;
        border: 1px solid rgba(27,42,74,0.18) !important;
    }}

    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {{
        color: {COLORS["text"]} !important;
        -webkit-text-fill-color: {COLORS["text"]} !important;
        font-size: 15px !important;
    }}

    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextArea"] label,
    div[data-testid="stTextArea"] label p,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] label p {{
        color: {COLORS["text"]} !important;
        font-size: 14px !important;
    }}

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {{
        color: {COLORS["text"]};
        font-size: 15px !important;
        line-height: 1.65 !important;
    }}

    /* Better contrast for alerts/messages */
    div[data-testid="stSuccess"] {{
        background-color: #D1E7DD !important;
        border: 1px solid #A3CFBB !important;
    }}
    div[data-testid="stSuccess"] p {{
        color: #0F5132 !important;
        font-weight: 600;
    }}

    div[data-testid="stInfo"] {{
        background-color: #CFF4FC !important;
        border: 1px solid #B6EFFB !important;
    }}
    div[data-testid="stInfo"] p {{
        color: #055160 !important;
        font-weight: 600;
    }}

    div[data-testid="stWarning"] {{
        background-color: #FFF3CD !important;
        border: 1px solid #FFECB5 !important;
    }}
    div[data-testid="stWarning"] p {{
        color: #664d03 !important;
        font-weight: 600;
    }}

    div[data-testid="stError"] {{
        background-color: #F8D7DA !important;
        border: 1px solid #F5C2C7 !important;
    }}
    div[data-testid="stError"] p {{
        color: #842029 !important;
        font-weight: 600;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: white;
        border-radius: 12px;
        padding: 0.3rem;
        box-shadow: 0 1px 3px rgba(27,42,74,0.08);
        gap: 0;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
        color: {COLORS["primary"]};
        padding: 0.6rem 1.2rem;
    }}

    .stTabs [aria-selected="true"] {{
        background: {COLORS["primary"]} !important;
        color: white !important;
    }}

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {{
        color: white !important;
    }}

    .doc-section h3,
    .doc-section h3:hover,
    .doc-section p,
    .doc-section p:hover,
    .doc-section li,
    .doc-section li:hover,
    .doc-section strong,
    .doc-section strong:hover {{
        color: {COLORS["text"]} !important;
    }}

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary:hover {{
        color: {COLORS["primary"]} !important;
        background-color: white !important;
    }}

    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary:hover span,
    div[data-testid="stExpander"] summary:hover p {{
        color: {COLORS["primary"]} !important;
    }}

    div[data-testid="stMarkdownContainer"] *:hover {{
        color: inherit !important;
    }}

    .result-card {{
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(27,42,74,0.1);
        border: 1px solid rgba(27,42,74,0.06);
        text-align: center;
        margin: 1rem 0;
    }}

    .result-approved {{
        border-left: 5px solid {COLORS["success"]};
    }}

    .result-refused {{
        border-left: 5px solid {COLORS["danger"]};
    }}

    .result-card .decision {{
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}

    .result-card .proba {{
        font-size: 1.05rem;
        color: {COLORS["muted"]};
        margin-top: 0.3rem;
    }}

    .info-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(27,42,74,0.06);
        font-size: 0.95rem;
    }}

    .info-row .info-label {{
        color: {COLORS["muted"]};
        font-weight: 500;
    }}

    .info-row .info-value {{
        color: {COLORS["primary"]};
        font-weight: 600;
    }}

    .doc-section {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(27,42,74,0.08);
        border: 1px solid rgba(27,42,74,0.06);
        color: {COLORS["primary"]};
        line-height: 1.7;
    }}

    .doc-section h3 {{
        color: {COLORS["primary"]};
        font-weight: 700;
        margin-top: 0;
        border-bottom: 2px solid {COLORS["accent"]};
        padding-bottom: 0.4rem;
    }}

    .doc-section p, .doc-section li {{
        color: {COLORS["text"]};
        font-size: 0.95rem;
    }}

    .doc-section code {{
        background: #E5E1D8;
        color: {COLORS["primary"]};
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: 500;
    }}

    .stSidebar {{
        background: white;
        border-right: 1px solid rgba(27,42,74,0.08);
    }}

    /* Sidebar text contrast */
    .stSidebar .stRadio > label,
    .stSidebar .stSelectbox > label,
    .stSidebar p,
    .stSidebar li {{
        color: {COLORS["text"]} !important;
    }}

    div[data-testid="stExpander"] {{
        background: white;
        border: 1px solid rgba(27,42,74,0.08);
        border-radius: 12px;
    }}

    div[data-testid="stExpander"] summary {{
        color: {COLORS["primary"]};
        font-weight: 600;
        font-size: 15px;
    }}

    /* Button text contrast */
    .stButton > button {{
        font-weight: 600;
        font-size: 15px;
    }}

    /* DataFrame/table styling */
    .stDataFrame {{
        font-size: 14px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <div class="main-header">
        <h1>Home Credit Risk Platform</h1>
        <p>Credit Scoring Model &mdash; Monitoring & Analytics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Load metadata ---
with open(ARTIFACTS_DIR / "model_metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = metadata.get("optimal_threshold", 0.494)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["chart_bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=14),
    hoverlabel=dict(bgcolor="white", font=dict(color=COLORS["text"], size=13)),
    margin=dict(l=50, r=20, t=50, b=50),
    xaxis=dict(
        gridcolor="rgba(27,42,74,0.08)",
        zerolinecolor="rgba(27,42,74,0.15)",
        title=dict(font=dict(color=COLORS["text"], size=13, weight=600)),
        tickfont=dict(color=COLORS["muted"], size=12),
        automargin=True,
        linewidth=1,
        linecolor="rgba(27,42,74,0.15)",
    ),
    yaxis=dict(
        gridcolor="rgba(27,42,74,0.08)",
        zerolinecolor="rgba(27,42,74,0.15)",
        title=dict(font=dict(color=COLORS["text"], size=13, weight=600)),
        tickfont=dict(color=COLORS["muted"], size=12),
        automargin=True,
        linewidth=1,
        linecolor="rgba(27,42,74,0.15)",
    ),
)

PLOTLY_LEGEND_STYLE = dict(
    bgcolor="rgba(255,255,255,0.88)",
    bordercolor="rgba(27,42,74,0.12)",
    borderwidth=1,
    font=dict(color=COLORS["text"], size=12),
)


@st.cache_resource
def load_model():
    """Charge le modele de scoring (cache Streamlit)."""
    import joblib

    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return model, feature_names


@st.cache_data(ttl=20, show_spinner=False)
def check_api_health(base_url: str, timeout_s: float = 2.5) -> dict:
    """Teste les endpoints principaux de l'API et retourne un statut synthétique."""
    base = base_url.rstrip("/")
    endpoints = {
        "health": "/health",
        "model_info": "/model-info",
    }

    results = {}
    for name, path in endpoints.items():
        url = f"{base}{path}"
        start = time.perf_counter()
        try:
            req = Request(url=url, method="GET")
            with urlopen(req, timeout=timeout_s) as resp:  # nosec: B310 - URL utilisateur attendue
                status_code = resp.getcode()
                elapsed_ms = (time.perf_counter() - start) * 1000
                body = resp.read()
                payload = {}
                if body:
                    try:
                        payload = json.loads(body.decode("utf-8"))
                    except Exception:
                        payload = {}
                results[name] = {
                    "ok": 200 <= status_code < 300,
                    "status_code": status_code,
                    "latency_ms": elapsed_ms,
                    "payload": payload,
                    "error": "",
                }
        except URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            results[name] = {
                "ok": False,
                "status_code": None,
                "latency_ms": elapsed_ms,
                "payload": {},
                "error": str(exc.reason) if hasattr(exc, "reason") else str(exc),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            results[name] = {
                "ok": False,
                "status_code": None,
                "latency_ms": elapsed_ms,
                "payload": {},
                "error": str(exc),
            }

    n_up = sum(1 for r in results.values() if r["ok"])
    api_up = results.get("health", {}).get("ok", False)
    return {
        "api_up": api_up,
        "n_up": n_up,
        "n_total": len(endpoints),
        "results": results,
    }


# === Documentation repliable (au-dessus des onglets) ===
with st.expander(
    "Guide du dashboard: contexte, objectifs et lecture des onglets", expanded=False
):
    st.markdown(
        f"""
        <div class="doc-section">
            <h3>Pourquoi cette application</h3>
            <p>
                Cette interface centralise le <strong>scoring credit</strong> et le
                <strong>monitoring ML</strong> pour suivre la qualite des decisions en production.
                L'objectif est de donner aux equipes metier et data un meme point d'observation:
                prediction client, distribution des scores, performance API et data drift.
            </p>
            <p>
                Contexte metier: le modele estime la probabilite de defaut. Si la probabilite est
                superieure au seuil de decision (<strong>{THRESHOLD:.3f}</strong>), la demande est
                classee <strong>REFUSED</strong>, sinon <strong>APPROVED</strong>.
            </p>
        </div>
        <div class="doc-section">
            <h3>Comment lire les onglets</h3>
            <ul>
                <li><strong>Prediction</strong> : simulation d'un scoring client (ID existant ou saisie manuelle).</li>
                <li><strong>Scores & Decisions</strong> : distribution des probabilites, taux de refus, volume de decisions.</li>
                <li><strong>Performance API</strong> : suivi de latence (moyenne, P50, P95, max) et tendance temporelle.</li>
                <li><strong>Data Drift</strong> : comparaison reference/production pour detecter les derivees de features.</li>
                <li><strong>Modele</strong> : recapitulatif du modele, du seuil optimal et des parametres metier.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# === TABS ===
tab_predict, tab_scores, tab_perf, tab_drift, tab_model = st.tabs(
    ["Prediction", "Scores & Decisions", "Performance API", "Data Drift", "Modele"]
)

# ============================================================
# TAB : Prediction client
# ============================================================
with tab_predict:
    st.markdown(
        '<div class="section-title">Scoring Client</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Simulez une **decision de credit** pour un client. Le modele calcule une probabilite "
        f"de defaut de paiement : si elle depasse le seuil de **{THRESHOLD:.3f}**, la demande est "
        f"refusee. La jauge visualise ou se situe le score par rapport au seuil."
    )

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown(
            "Entrez un **identifiant client** pour charger ses features depuis les donnees "
            "de test, ou saisissez des features manuellement."
        )

        input_mode = st.radio(
            "Mode de saisie",
            ["Par identifiant client", "Manuel (features)"],
            horizontal=True,
        )

        client_features = {}
        client_id = 0

        if input_mode == "Par identifiant client":
            SAMPLE_CLIENT_IDS = [
                100001, 100005, 100013, 100028, 100038,
                100042, 100057, 100065, 100066, 100067,
                100068, 100071, 100074, 100075, 100077,
            ]
            id_choice = st.selectbox(
                "Identifiant client",
                options=SAMPLE_CLIENT_IDS,
                format_func=lambda x: f"Client #{x}",
                help="15 clients pré-sélectionnés depuis le jeu de test",
            )
            use_custom = st.checkbox("Saisir un identifiant personnalisé")
            if use_custom:
                client_id = st.number_input(
                    "SK_ID_CURR personnalisé", min_value=0, value=id_choice, step=1
                )
            else:
                client_id = id_choice
            if REFERENCE_DATA.exists():
                ref_full = pd.read_csv(REFERENCE_DATA, nrows=5000)
                if "SK_ID_CURR" in ref_full.columns:
                    client_row = ref_full[ref_full["SK_ID_CURR"] == client_id]
                    if len(client_row) > 0:
                        client_features = (
                            client_row.drop("SK_ID_CURR", axis=1).iloc[0].to_dict()
                        )
                        st.success(
                            f"Client {client_id} trouve — {len(client_features)} features chargees"
                        )
                    else:
                        st.warning(
                            f"Client {client_id} introuvable dans les 5 000 premiers clients de test."
                        )
            else:
                st.warning("Fichier test_preprocessed.csv non disponible.")
        else:
            st.markdown(
                "Saisissez les features au format `NOM: valeur` (une par ligne) :"
            )
            manual_input = st.text_area(
                "Features (une par ligne)",
                value="AMT_CREDIT: 0.5\nAMT_ANNUITY: -0.3\nEXT_SOURCE_2: 0.7",
                height=150,
            )
            for line in manual_input.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    try:
                        client_features[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
            if client_features:
                st.info(f"{len(client_features)} features saisies")

        predict_btn = st.button(
            "Lancer le scoring", type="primary", use_container_width=True
        )

    with col_result:
        if predict_btn and client_features:
            _api_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
            sk_id = client_id if input_mode == "Par identifiant client" else 0

            api_used = False
            try:
                import json as _json
                from urllib.request import Request as _Req, urlopen as _urlopen

                _payload = _json.dumps(
                    {"SK_ID_CURR": sk_id, "features": client_features}
                ).encode()
                _req = _Req(
                    url=f"{_api_url}/predict",
                    data=_payload,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with _urlopen(_req, timeout=5) as _resp:  # nosec: B310
                    _result = _json.loads(_resp.read().decode())
                proba = _result["probability"]
                elapsed_ms = _result["inference_time_ms"]
                prediction = _result["prediction"]
                decision = _result["decision"]
                api_used = True
            except Exception:
                # Fallback: inférence locale si l'API est indisponible
                model, feature_names = load_model()
                start = time.perf_counter()
                df = pd.DataFrame([client_features])
                df = df.reindex(columns=feature_names, fill_value=0)
                proba = float(model.predict_proba(df)[:, 1][0])
                elapsed_ms = (time.perf_counter() - start) * 1000
                prediction = int(proba >= THRESHOLD)
                decision = "REFUSED" if prediction == 1 else "APPROVED"

            if not api_used:
                st.warning(
                    f"API indisponible sur {_api_url} — inférence locale (prédiction non enregistrée)."
                )
            color = COLORS["danger"] if prediction == 1 else COLORS["success"]
            css_class = "result-refused" if prediction == 1 else "result-approved"

            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div class="decision" style="color: {color};">{decision}</div>
                    <div class="proba">Probabilite de defaut : <strong>{proba:.4f}</strong></div>
                    <div class="proba">Seuil : {THRESHOLD:.3f} &nbsp;|&nbsp; Inference : {elapsed_ms:.1f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba,
                    number=dict(
                        font=dict(size=36, color=COLORS["primary"]),
                        valueformat=".4f",
                    ),
                    gauge=dict(
                        axis=dict(range=[0, 1], tickfont=dict(size=12)),
                        bar=dict(color=color, thickness=0.3),
                        bgcolor="white",
                        steps=[
                            dict(range=[0, THRESHOLD], color="rgba(29,106,75,0.1)"),
                            dict(range=[THRESHOLD, 1], color="rgba(139,45,45,0.1)"),
                        ],
                        threshold=dict(
                            line=dict(color=COLORS["accent"], width=3),
                            thickness=0.8,
                            value=THRESHOLD,
                        ),
                    ),
                )
            )
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=30, r=30, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        elif predict_btn:
            st.warning("Aucune feature chargee. Verifiez l'identifiant ou la saisie.")

# ============================================================
# TAB : Distribution des scores
# ============================================================
with tab_scores:
    st.markdown(
        "Vue d'ensemble des **predictions passees** enregistrees par l'API. "
        "L'histogramme montre comment les scores de probabilite se repartissent entre "
        "clients approuves (en vert, sous le seuil) et refuses (en rouge, au-dessus). "
        "Le camembert resume la proportion globale, et la courbe temporelle permet de "
        "detecter des variations de volume inhabituelles."
    )
    # Essai DB d'abord (prod), puis JSONL (local)
    logs = load_predictions_from_db()
    if logs is None and PREDICTIONS_LOG.exists():
        logs = pd.read_json(PREDICTIONS_LOG, lines=True)

    if logs is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predictions", f"{len(logs):,}")
        col2.metric("Taux de refus", f"{logs['prediction'].mean():.1%}")
        col3.metric("Score moyen", f"{logs['probability'].mean():.3f}")
        col4.metric("Seuil", f"{THRESHOLD:.3f}")

        st.markdown(
            '<div class="section-title">Distribution des Scores</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        approved = logs[logs["prediction"] == 0]["probability"]
        refused = logs[logs["prediction"] == 1]["probability"]

        fig.add_trace(
            go.Histogram(
                x=approved,
                nbinsx=40,
                name="Approved",
                marker_color=COLORS["approved"],
                opacity=0.75,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=refused,
                nbinsx=40,
                name="Refused",
                marker_color=COLORS["refused"],
                opacity=0.75,
            )
        )
        fig.add_vline(
            x=THRESHOLD,
            line_dash="dash",
            line_color=COLORS["accent"],
            line_width=2,
            annotation_text=f"  Seuil ({THRESHOLD:.3f})",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["text"], size=12),
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Probabilite de defaut",
            yaxis_title="Nombre de predictions",
            legend=dict(
                **PLOTLY_LEGEND_STYLE,
                orientation="h",
                y=1.12,
                x=0.5,
                xanchor="center",
            ),
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                '<div class="section-title">Repartition</div>',
                unsafe_allow_html=True,
            )
            n_approved = int((logs["prediction"] == 0).sum())
            n_refused = int((logs["prediction"] == 1).sum())
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Approved", "Refused"],
                        values=[n_approved, n_refused],
                        marker=dict(colors=[COLORS["approved"], COLORS["refused"]]),
                        hole=0.55,
                        textinfo="percent+label",
                        textfont=dict(size=13, color=COLORS["text"]),
                        insidetextorientation="horizontal",
                    )
                ]
            )
            fig_pie.update_layout(
                showlegend=False,
                height=350,
                **{
                    k: v
                    for k, v in PLOTLY_LAYOUT.items()
                    if k not in ("xaxis", "yaxis")
                },
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.markdown(
                '<div class="section-title">Volume temporel</div>',
                unsafe_allow_html=True,
            )
            logs["timestamp"] = pd.to_datetime(logs["timestamp"])
            logs_hourly = (
                logs.set_index("timestamp")
                .resample("h")["prediction"]
                .agg(["count", "mean"])
            )
            if len(logs_hourly) > 1:
                fig_time = go.Figure()
                fig_time.add_trace(
                    go.Scatter(
                        x=logs_hourly.index,
                        y=logs_hourly["count"],
                        mode="lines+markers",
                        name="Volume",
                        line=dict(color=COLORS["secondary"], width=2),
                        marker=dict(size=5),
                    )
                )
                fig_time.update_layout(
                    xaxis_title="",
                    yaxis_title="Predictions / heure",
                    height=350,
                    showlegend=False,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("Donnees temporelles insuffisantes.")
    else:
        st.info("En attente de predictions. Lancez l'API et envoyez des requetes.")

# ============================================================
# TAB : Performance API
# ============================================================
with tab_perf:
    st.markdown(
        "Cet onglet verifie en temps reel si l'**API de scoring** est operationnelle. "
        "Il teste les endpoints `/health` et `/model-info`, puis affiche les metriques de "
        "**latence d'inference** (temps que met le modele a repondre). "
        "P50 = moitie des requetes sont plus rapides, P95 = 95% des requetes sont plus rapides. "
        "Si l'API n'est pas lancee, les statuts afficheront KO — c'est normal."
    )
    st.markdown(
        '<div class="section-title">Sante API & Endpoints</div>',
        unsafe_allow_html=True,
    )
    api_col, timeout_col = st.columns([3, 1])
    with api_col:
        _default_api_url = os.environ.get("API_URL", "http://localhost:8000")
        api_base_url = st.text_input(
            "URL de l'API a monitorer",
            value=_default_api_url,
            help="Configuré via la variable d'env API_URL",
        )
    with timeout_col:
        timeout_s = st.number_input(
            "Timeout (s)",
            min_value=1.0,
            max_value=10.0,
            value=2.5,
            step=0.5,
        )

    health = check_api_health(api_base_url, timeout_s=timeout_s)
    health_result = health["results"].get("health", {})
    model_result = health["results"].get("model_info", {})
    api_status = "UP" if health["api_up"] else "DOWN"
    api_latency = health_result.get("latency_ms")

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric(
        "Statut API",
        api_status,
        f"{api_latency:.0f} ms" if isinstance(api_latency, (int, float)) else "n/a",
    )
    col_h2.metric(
        "Endpoint /health",
        "OK" if health_result.get("ok") else "KO",
        (
            f"HTTP {health_result.get('status_code')}"
            if health_result.get("status_code") is not None
            else "indisponible"
        ),
    )
    col_h3.metric(
        "Endpoint /model-info",
        "OK" if model_result.get("ok") else "KO",
        (
            f"HTTP {model_result.get('status_code')}"
            if model_result.get("status_code") is not None
            else "indisponible"
        ),
    )
    col_h4.metric("Endpoints UP", f"{health['n_up']}/{health['n_total']}")

    if health["api_up"]:
        payload = health_result.get("payload", {})
        model_loaded = payload.get("model_loaded")
        n_features = payload.get("n_features")
        threshold = payload.get("threshold")
        st.success(
            "API operationnelle"
            f" | model_loaded={model_loaded}"
            f" | n_features={n_features}"
            f" | threshold={threshold}"
        )
    else:
        err = health_result.get("error") or "Connexion impossible"
        st.warning(f"API indisponible sur {api_base_url}. Detail: {err}")

    st.markdown(
        '<div class="section-title">Performance des predictions (logs)</div>',
        unsafe_allow_html=True,
    )

    logs = load_predictions_from_db()
    if logs is None and PREDICTIONS_LOG.exists():
        logs = pd.read_json(PREDICTIONS_LOG, lines=True)

    if logs is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latence moyenne", f"{logs['inference_time_ms'].mean():.1f} ms")
        col2.metric("Latence P50", f"{logs['inference_time_ms'].median():.1f} ms")
        col3.metric("Latence P95", f"{logs['inference_time_ms'].quantile(0.95):.1f} ms")
        col4.metric("Latence max", f"{logs['inference_time_ms'].max():.1f} ms")

        st.markdown(
            '<div class="section-title">Distribution de la Latence</div>',
            unsafe_allow_html=True,
        )

        fig_lat = go.Figure()
        fig_lat.add_trace(
            go.Histogram(
                x=logs["inference_time_ms"],
                nbinsx=40,
                marker_color=COLORS["secondary"],
                opacity=0.8,
            )
        )
        p95 = logs["inference_time_ms"].quantile(0.95)
        fig_lat.add_vline(
            x=p95,
            line_dash="dash",
            line_color=COLORS["danger"],
            annotation_text=f"  P95 ({p95:.1f} ms)",
            annotation_position="top right",
            annotation_font=dict(size=12, color=COLORS["text"]),
        )
        fig_lat.update_layout(
            xaxis_title="Temps d'inference (ms)",
            yaxis_title="Nombre de requetes",
            showlegend=False,
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        st.markdown(
            '<div class="section-title">Latence dans le temps</div>',
            unsafe_allow_html=True,
        )

        logs["timestamp"] = pd.to_datetime(logs["timestamp"])
        mean_lat = logs["inference_time_ms"].mean()
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=logs["timestamp"],
                y=logs["inference_time_ms"],
                mode="markers",
                marker=dict(color=COLORS["secondary"], size=4, opacity=0.4),
            )
        )
        fig_ts.add_hline(
            y=mean_lat,
            line_dash="dash",
            line_color=COLORS["accent"],
            annotation_text=f"  Moyenne : {mean_lat:.1f} ms",
            annotation_position="top left",
            annotation_font=dict(size=12, color=COLORS["text"]),
        )
        fig_ts.update_layout(
            xaxis_title="",
            yaxis_title="Latence (ms)",
            showlegend=False,
            height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info(
            "En attente de donnees de performance (fichier predictions_log.jsonl absent)."
        )

# ============================================================
# TAB : Data Drift
# ============================================================
with tab_drift:
    st.markdown(
        "Le **data drift** mesure si les donnees en production different des donnees d'entrainement. "
        "Si les features changent significativement, le modele risque de perdre en fiabilite. "
        "Ici on utilise le **test de Kolmogorov-Smirnov (KS)** : plus la statistique KS est elevee, "
        "plus la distribution a change. Les barres rouges indiquent un drift detecte (p-value < 0.05). "
        "Utilisez la sidebar a gauche pour simuler differents types de drift et observer leur impact."
    )
    if REFERENCE_DATA.exists():
        st.sidebar.markdown("### Simulation du Drift")

        drift_type = st.sidebar.selectbox(
            "Type de drift",
            ["none", "gradual", "sudden", "feature_shift"],
            index=1,
            format_func=lambda x: {
                "none": "Aucun drift",
                "gradual": "Drift graduel",
                "sudden": "Drift soudain",
                "feature_shift": "Shift de features",
            }[x],
        )

        DRIFT_EXPLANATIONS = {
            "none": (
                "**Aucun drift** — Les données simulées sont identiques aux données "
                "d'entraînement. Le modèle devrait performer normalement."
            ),
            "gradual": (
                "**Drift graduel** — Bruit gaussien progressivement ajouté sur toutes "
                "les features. Simule une dérive lente et naturelle des données (vieillissement "
                "du portefeuille, inflation, évolution des comportements)."
            ),
            "sudden": (
                "**Drift soudain** — Décalage brutal sur les 20 features les plus "
                "importantes. Simule un choc externe : nouvelle réglementation, crise "
                "économique, changement de segment clientèle."
            ),
            "feature_shift": (
                "**Shift de features** — Mise à l'échelle multiplicative sur des features "
                "aléatoires. Simule un problème de collecte ou de transformation des données "
                "(bug de pipeline, changement de source)."
            ),
        }
        st.sidebar.info(DRIFT_EXPLANATIONS[drift_type])

        intensity = st.sidebar.slider("Intensite", 0.0, 1.0, 0.3, 0.05)
        n_samples = st.sidebar.slider("Echantillons", 100, 5000, 1000, 100)

        ref_data = pd.read_csv(REFERENCE_DATA, nrows=n_samples)
        if "SK_ID_CURR" in ref_data.columns:
            ref_data = ref_data.drop("SK_ID_CURR", axis=1)

        prod_data = simulate_drift(ref_data, drift_type=drift_type, intensity=intensity)
        drift_report = compute_drift_report(ref_data, prod_data, top_n=20)

        n_drifted = int(drift_report["drift_detected"].sum())
        n_total = len(drift_report)
        drift_pct = n_drifted / n_total * 100 if n_total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Features analysees", n_total)
        col2.metric("Features en drift", n_drifted)
        col3.metric("Taux de drift", f"{drift_pct:.0f}%")

        if drift_pct > 30:
            st.markdown(
                '<span class="status-badge status-alert">ALERTE — Drift significatif detecte</span>',
                unsafe_allow_html=True,
            )
        elif drift_pct > 0:
            st.markdown(
                '<span class="status-badge status-ok">Drift modere — Surveillance recommandee</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-ok">Aucun drift detecte</span>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-title">Top Features par Drift (KS Statistic)</div>',
            unsafe_allow_html=True,
        )

        # Tronquer les noms longs pour la lisibilite du graphe
        display_report = drift_report.copy()
        display_report["feature_short"] = display_report["feature"].apply(
            lambda x: x[:28] + "..." if len(x) > 28 else x
        )

        fig_ks = go.Figure()
        fig_ks.add_trace(
            go.Bar(
                y=display_report["feature_short"],
                x=display_report["ks_statistic"],
                orientation="h",
                marker_color=[
                    COLORS["danger"] if d else COLORS["secondary"]
                    for d in display_report["drift_detected"]
                ],
                opacity=0.85,
                text=[f"{v:.3f}" for v in display_report["ks_statistic"]],
                textposition="outside",
                textfont=dict(size=11, color=COLORS["text"]),
                hovertext=display_report["feature"],
            )
        )
        fig_ks.update_layout(
            margin=dict(l=210, r=40, t=50, b=40),
            yaxis=dict(
                autorange="reversed",
                gridcolor="rgba(27,42,74,0.06)",
                tickfont=dict(size=11, color=COLORS["text"]),
                automargin=True,
            ),
            xaxis_title="KS Statistic",
            height=max(400, n_total * 28),
            showlegend=False,
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("yaxis", "margin")},
        )
        st.plotly_chart(fig_ks, use_container_width=True)

        # Distributions comparees top 3
        top_drifted = drift_report[drift_report["drift_detected"]].head(3)
        if len(top_drifted) > 0:
            st.markdown(
                '<div class="section-title">Distributions Comparees (Reference vs Production)</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(top_drifted), 3))
            for idx, (_, row) in enumerate(top_drifted.iterrows()):
                feat = row["feature"]
                with cols[idx]:
                    fig_comp = go.Figure()
                    fig_comp.add_trace(
                        go.Histogram(
                            x=ref_data[feat],
                            name="Reference",
                            opacity=0.6,
                            marker_color=COLORS["chart_ref"],
                            nbinsx=30,
                        )
                    )
                    fig_comp.add_trace(
                        go.Histogram(
                            x=prod_data[feat],
                            name="Production",
                            opacity=0.6,
                            marker_color=COLORS["chart_prod"],
                            nbinsx=30,
                        )
                    )
                    short_name = feat[:20] + "..." if len(feat) > 20 else feat
                    fig_comp.update_layout(
                        title=dict(
                            text=(
                                f"<b>{short_name}</b><br>"
                                f"<span style='font-size:11px;color:#8B95A5;'>"
                                f"KS = {row['ks_statistic']:.3f}</span>"
                            ),
                            font=dict(size=13, color=COLORS["text"]),
                        ),
                        barmode="overlay",
                        height=300,
                        showlegend=idx == 0,
                        legend=dict(
                            **{k: v for k, v in PLOTLY_LEGEND_STYLE.items() if k != "font"},
                            orientation="h",
                            y=1.2,
                            font=dict(size=11, color=COLORS["text"]),
                        ),
                        margin=dict(l=30, r=10, t=70, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor=COLORS["chart_bg"],
                        hoverlabel=dict(
                            bgcolor="white", font=dict(color=COLORS["text"])
                        ),
                        font=dict(
                            family="Inter, sans-serif", size=11, color=COLORS["text"]
                        ),
                        xaxis=dict(
                            title=dict(font=dict(color=COLORS["text"])),
                        ),
                        yaxis=dict(
                            title=dict(font=dict(color=COLORS["text"])),
                            tickfont=dict(color=COLORS["muted"], size=10),
                        ),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

        # --- Rapport Evidently AI ---
        st.markdown(
            '<div class="section-title">Rapport Evidently AI</div>',
            unsafe_allow_html=True,
        )

        evidently_mode = st.radio(
            "Donnees a analyser",
            ["Reference (train) vs Test reel", "Reference vs Drift simule (ci-dessus)"],
            horizontal=True,
            help="Le mode 'Test reel' compare les donnees d'entrainement aux donnees de test."
            " Le mode 'Drift simule' utilise la simulation configuree dans la sidebar.",
        )

        if st.button("Generer le rapport Evidently", type="primary"):
            with st.spinner("Generation du rapport Evidently..."):
                import webbrowser

                if evidently_mode.startswith("Reference (train) vs Test"):
                    # Charger les vraies donnees de test
                    test_data = pd.read_csv(
                        Path("data/test_preprocessed.csv"), nrows=n_samples
                    )
                    if "SK_ID_CURR" in test_data.columns:
                        test_data = test_data.drop("SK_ID_CURR", axis=1)
                    # Colonnes communes entre ref et test
                    common = sorted(
                        set(ref_data.columns) & set(test_data.columns)
                    )
                    ev_ref = ref_data[common]
                    ev_cur = test_data[common]
                else:
                    # Utiliser les donnees simulees
                    ev_ref = ref_data
                    ev_cur = prod_data

                evidently_report = Report([DataDriftPreset()])
                evidently_snapshot = evidently_report.run(ev_ref, ev_cur)

                report_path = Path("monitoring/drift_report_evidently.html")
                evidently_snapshot.save_html(str(report_path))
                webbrowser.open(f"file://{report_path.resolve()}")
                st.success(
                    f"Rapport ouvert dans le navigateur ({report_path.resolve()})"
                )
    else:
        st.info("Fichier de reference non disponible.")

# ============================================================
# TAB : Info Modele
# ============================================================
with tab_model:
    st.markdown(
        "Recapitulatif du modele deploye et de ses parametres metier. "
        "Le **seuil optimal** a ete calibre pour minimiser le cout metier : un faux negatif "
        "(client defaillant classe comme bon) coute plus cher qu'un faux positif "
        "(bon client refuse a tort). Le ratio des couts FN/FP reflete cette asymetrie."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-title">Performance du Modele</div>',
            unsafe_allow_html=True,
        )
        info_perf = [
            ("Modele", metadata.get("best_model_name", "N/A")),
            ("Seuil optimal", f"{metadata.get('optimal_threshold', 0):.4f}"),
            (
                "Business Cost (optimal)",
                f"{metadata.get('business_cost_optimal', 0):.4f}",
            ),
            ("Nombre de features", str(metadata.get("n_features", "N/A"))),
        ]
        for label, value in info_perf:
            st.markdown(
                f'<div class="info-row">'
                f'<span class="info-label">{label}</span>'
                f'<span class="info-value">{value}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            '<div class="section-title">Parametres Metier</div>',
            unsafe_allow_html=True,
        )
        dist = metadata.get("class_distribution", {})
        total = sum(int(v) for v in dist.values()) if dist else 0
        default_rate = int(dist.get("1", 0)) / total if total > 0 else 0

        info_biz = [
            ("Cout Faux Negatif (FN)", f"{metadata.get('cost_fn', 'N/A')}x"),
            ("Cout Faux Positif (FP)", f"{metadata.get('cost_fp', 'N/A')}x"),
            (
                "Echantillon d'entrainement",
                f"{metadata.get('n_train_samples', 0):,} clients",
            ),
            ("Taux de defaut historique", f"{default_rate:.2%}"),
        ]
        for label, value in info_biz:
            st.markdown(
                f'<div class="info-row">'
                f'<span class="info-label">{label}</span>'
                f'<span class="info-value">{value}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    with st.expander("Configuration complete (JSON)"):
        st.json(metadata)
