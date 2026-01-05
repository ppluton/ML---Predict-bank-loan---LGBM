# 🏦 Modèle de Scoring de Crédit - Projet MLOps OpenClassrooms

## Formation AI Engineer 2026 - Projet OC6

[![MLFlow](https://img.shields.io/badge/MLFlow-Tracking-blue.svg)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit-learn-ML-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-yellow.svg)](https://lightgbm.readthedocs.io/)

### 📊 **Résumé Exécutif**

**Problème métier** : Prédire le risque de défaut de paiement des clients d'une institution financière de microcrédit (Home Credit Default Risk).

**Défi principal** : Dataset massivement déséquilibré (91.9% bons clients vs 8.1% défauts → ratio **11.4:1**) + 8 tables relationnelles à agréger.

**Solution proposée** : Pipeline MLOps complet avec **innovations méthodologiques** :

- Agrégation hiérarchique de 57M+ lignes → 305 features
- **Feature "Has_History"** : capture l'absence d'historique (info critique)
- **Imputation stratégique** : 5 approches selon sémantique métier
- **Score métier personnalisé** : FN = 10× FP (priorité recall - consigne OpenClassrooms)
- **MLFlow tracking complet** : baselines, tuning, seuil optimal

**Résultats** : Modèle LightGBM optimisé avec coût métier minimisé, prêt pour production.

---

## 🎯 **Objectifs du Projet**

1. **Ingénierie des features avancées** à partir de données relationnelles complexes
2. **Pipeline preprocessing robuste** gérant intelligemment les NaN métier
3. **Modélisation orientée business** avec score coût asymétrique
4. **MLOps** : tracking expérimentations, reproductibilité, model registry
5. **Optimisation du seuil de décision** pour maximiser le recall métier

---

## 🏗️ **Architecture du Pipeline MLOps**

```
📥 Données Brutes (8 CSV)
    ↓ Agrégation Hiérarchique (Notebook 01)
📊 train_aggregated.csv (307k × 305 features)
    ↓ Preprocessing + Feature Engineering (Notebook 02)
⚙️ train_preprocessed.csv (307k × 265 features, 0 NaN, scalé)
    ↓ Modeling + MLFlow (Notebook 03)
🚀 Meilleur Modèle LightGBM (tracké MLFlow)
    ↓ Seuil Optimal + Production Ready
📤 submission.csv (prédictions Kaggle)
```

---

## 📁 **Structure du Projet**

```
OC6_MLOPS/
├── data/                          # Données brutes et traitées
│   ├── application_train.csv      # Table principale (307k lignes)
│   ├── bureau.csv                 # Historique crédits externes (1.7M)
│   ├── train_aggregated.csv       # Après Notebook 01 (305 features)
│   ├── train_preprocessed.csv     # Après Notebook 02 (265 features)
│   └── submission.csv             # Prédictions finales
├── notebooks/                     # Pipeline en 3 étapes
│   ├── 01_EDA.ipynb               # EDA + Agrégation
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   └── 03_modeling_with_MLFLOW.ipynb
├── notebooks/charts_eda/          # Visualisations EDA
│   ├── graphique_1_age_distribution.png
│   ├── graphique_2_correlations.png
│   └── graphique_5_historique_bureau.png
├── src/                           # Code modulaire (production-ready)
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── metrics.py
│   └── modeling.py
├── mlruns/                        # MLFlow tracking automatique
├── models/                        # Modèles sauvegardés
├── pyproject.toml                 # Dépendances (uv/pip)
├── uv.lock                        # Lockfile uv
└── README.md                      # Ce fichier
```

---

## 🚀 **Installation & Exécution**

### Prérequis

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommandé) ou pip

### Installation

```bash
git clone <votre-repo>
cd OC6_MLOPS
uv sync          # ou pip install -e .
```

### Lancer le Pipeline Complet

```bash
# 1. EDA + Agrégation
jupyter notebook notebooks/01_EDA.ipynb

# 2. Preprocessing + Features
jupyter notebook notebooks/02_preprocessing_and_feature_engineering.ipynb

# 3. Modeling + MLFlow
jupyter notebook notebooks/03_modeling_with_MLFLOW.ipynb

# Visualiser les expériences
mlflow ui          # http://localhost:5000
```

**Dépendances principales** : `pandas`, `scikit-learn`, `lightgbm`, `mlflow`, `matplotlib`, `seaborn`, `joblib`

---

## 📝 **Méthodologie Détaillée par Notebook**

### **Notebook 01 : EDA & Agrégation Hiérarchique** 🔍

**Objectifs** :

- Charger 8 tables (57M+ lignes total)
- Analyser relations : `application_train ← bureau ← bureau_balance`, `previous_application ← POS/CC/Installments`
- Créer dataset plat pour ML

**Innovations** :

- **Agrégation en cascade** : `bureau_balance` (27M) → `bureau` → client
- 183 features créées : 45 bureau + 138 previous_application
- **Statistiques riches** : min/max/mean/sum + one-hot catégorielles
- **Visualisations avancées** : 5 graphiques EDA (âge, corrélations, EXT_SOURCE, ratios, bureau)

**Résultats** :

```
307,511 clients × 305 features
Déséquilibre : 91.9% bons vs 8.1% défauts (11.4:1)
250/305 colonnes NaN (normal : absence historique)
Outputs : train_aggregated.csv + test_aggregated.csv
```

### **Notebook 02 : Preprocessing & Feature Engineering Avancé** ⚙️

**Objectifs** :

- Gérer 250 colonnes NaN intelligemment
- Créer features métier prédictives
- Préparer données scalées pour ML

**🚀 Innovations Clés** :

1. **Feature "Has_History" (INNOVATION PROPRIA)** :

   ```
   HAS_BUREAU, HAS_PREV_APP, HAS_CREDIT_CARD, HAS_POS_CASH, HAS_INSTALLMENTS
   Créées AVANT imputation → capture "aucun historique = info métier"
   ```

2. **Imputation Stratégique (5 règles sémantiques)** :
   | Type Colonne | Stratégie | Exemple | Rationale |
   |------------------|---------------|--------------------------|-----------|
   | Montants (AMT*) | 0 | AMT_CREDIT_SUM → 0 | Pas de crédit = 0€ |
   | Comptages (CNT*) | 0 | SK_ID_BUREAU_COUNT → 0 | 0 occurrence |
   | Dates (DAYS*) | -999 | DAYS_BIRTH → -999 | Sentinelle |
   | Moyennes (MEAN*) | Médiane | EXT_SOURCE_MEAN → median | Robuste outliers |
   | Autres | Médiane | - | Défaut conservateur |

3. **Feature Engineering Métier (11 nouvelles)** :
   ```
   💰 CREDIT_INCOME_RATIO (règle 33%)
   💳 ANNUITY_INCOME_RATIO (capacité remboursement)
   👴 AGE_YEARS, 👷 EMPLOYMENT_YEARS
   📊 EXT_SOURCE_MEAN/PROD (scores agrégés)
   👨‍👩‍👧 INCOME_PER_PERSON, CHILDREN_RATIO
   🏦 BUREAU_DEBT_INCOME_RATIO
   ```

**Résultats** :

```
307k × 265 features | 0 NaN | 0 Inf | Scalé (mean=0, std=1)
-45 colonnes (>80% NaN supprimées)
Scaler.pkl sauvegardé (production-ready)
```

### **Notebook 03 : Modeling MLOps avec MLFlow** 🎯

**Objectifs** :

- Baselines + tuning
- Score métier asymétrique
- Tracking reproductible

**🚀 Innovations** :

1. **Score Métier Personnalisé** (consigne OpenClassrooms) :

   ```python
   coût_total = (FN × 10) + FP    # Recall prioritaire
   ```

2. **3 Baselines Comparées** :
   | Modèle | Avantages | CV Business Cost |
   |------------------|------------------------|------------------|
   | Logistic Reg | Linéaire, rapide | Baseline |
   | Random Forest | Non-linéaire | Moyen |
   | **LightGBM** | **Gradient Boosting** | **Meilleur** |

3. **Hyperparameter Tuning** : GridSearchCV (27 combinaisons)
4. **Seuil Optimal** : ~0.3-0.4 (vs 0.5 défaut) → +X% recall
5. **MLFlow Complet** :
   - Paramètres, métriques CV/train
   - Matrices confusion visualisées
   - Modèles loggés + artifacts

**Outputs** :

```
submission.csv (Kaggle-ready)
mlruns/ (tracking)
model_metadata.json
```

---

## 💡 **Points Forts Méthodologiques (Jury)**

| Innovation                | Impact Métier/Business                     |
| ------------------------- | ------------------------------------------ |
| **Has_History features**  | "Nouveau client" = risque → info critique  |
| **Imputation sémantique** | Respecte logique bancaire (0€=pas crédit)  |
| **Score FN=10×FP**        | Recall prioritaire (perte >> manque gain)  |
| **Seuil optimisé**        | +X% performance coût métier                |
| **No Data Leakage**       | Scaler fit train only                      |
| **MLFlow end-to-end**     | Reproductible, auditable, production-ready |

**Gestion Déséquilibre** : `class_weight=balanced` + score asymétrique + seuil optimisé.

---

## 📊 **Métriques Clés (Placeholders - à finaliser)**

```
Dataset : 307k train | 48k test | 11.4:1 imbalance
Features: 122 orig → 305 agrégées → 265 finales
NaN : 82% → 0%
Meilleur Modèle : LightGBM Tuned
CV Business Cost : [X.XX] ± [X.XX]
Train AUC : [XX.X]%
Seuil Optimal : [X.XX] (vs 0.5)
Amélioration seuil : [+X.X]%
```
---

## 👨‍💻 **Auteur & Licence**

**Auteur** : Pierre Pluton  
**Formation** : OpenClassrooms AI Engineer 2026 - Projet OC6 MLOps  
**Date** : Janvier 2026

**Licence** : MIT License

```
© 2026 Pierre Pluton. Tous droits réservés pour OpenClassrooms.
```

---

**Merci d'avoir reviewé ce projet !** 🎉  
**Contact** : [votre-email] | [LinkedIn/GitHub]
