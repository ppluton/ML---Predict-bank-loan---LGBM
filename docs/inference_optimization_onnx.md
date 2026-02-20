# Fiche pédagogique — Optimisation d'inférence (ONNX)

> **Contexte** : Cette fiche explique pourquoi et comment le modèle LightGBM a été exporté au format ONNX, comment le profiling a révélé le vrai goulot d'étranglement, et comment l'API bascule automatiquement entre les deux moteurs d'inférence. Résultat mesuré : **57× plus rapide** en latence P50 single-row.

---

## Table des matières

1. [Le problème : qu'est-ce que la latence d'inférence ?](#1-le-problème--quest-ce-que-la-latence-dinférence)
2. [Profiling : trouver le vrai goulot d'étranglement](#2-profiling--trouver-le-vrai-goulot-détranglement)
3. [ONNX : le format universel des modèles ML](#3-onnx--le-format-universel-des-modèles-ml)
4. [La conversion LightGBM → ONNX avec onnxmltools](#4-la-conversion-lightgbm--onnx-avec-onnxmltools)
5. [ONNX Runtime : exécuter le modèle sans framework ML](#5-onnx-runtime--exécuter-le-modèle-sans-framework-ml)
6. [Le dual-path dans l'API : ONNX d'abord, LightGBM en fallback](#6-le-dual-path-dans-lapi--onnx-dabord-lightgbm-en-fallback)
7. [Validation numérique : vérifier que les deux modèles sont équivalents](#7-validation-numérique--vérifier-que-les-deux-modèles-sont-équivalents)
8. [Benchmark : mesurer le gain réel](#8-benchmark--mesurer-le-gain-réel)
9. [Vue d'ensemble et résultats](#9-vue-densemble-et-résultats)

---

## 1. Le problème : qu'est-ce que la latence d'inférence ?

### L'analogie du restaurant

Imaginez un restaurant où chaque plat demande 10 étapes : recevoir la commande, sortir les ingrédients du frigo, préparer les légumes, cuire, dresser l'assiette, etc. Si vous découvrez que **80% du temps est passé à chercher les ingrédients dans le frigo** (et non à cuire), l'optimisation évidente est de garder les ingrédients à portée de main — pas d'acheter un four plus rapide.

En ML, c'est pareil. La **latence d'inférence** est le temps écoulé entre "je reçois les features d'un client" et "je retourne la probabilité de défaut". Elle se décompose en plusieurs sous-étapes, et le goulot peut être n'importe laquelle d'entre elles.

### Ce que "single-row inference" signifie

Dans ce projet, chaque appel `/predict` prédit pour **un seul client** (pas un batch de 10 000). C'est le cas typique des APIs de scoring temps réel :

```
Client → POST /predict → 1 client → probabilité → réponse en < 100ms
```

C'est différent de l'inférence batch (nuit, 300k clients), où la vectorisation est reine. En single-row, les overheads fixes dominent.

---

## 2. Profiling : trouver le vrai goulot d'étranglement

### Qu'est-ce que le profiling ?

Le profiling, c'est **mesurer où le temps est réellement dépensé** dans un programme. Sans profiling, on optimise au doigt mouillé. Avec, on sait exactement quelle ligne de code prend 90% du temps.

```
Sans profiling           Avec profiling
───────────────          ────────────────────────────────────
"Je pense que            "cProfile dit :
 predict_proba           → DataFrame construction : 0.52 ms (73%)
 est lent"               → predict_proba           : 0.16 ms (23%)
                         → sérialisation dict      : 0.03 ms  (4%)
                         ↓
                         L'optimisation évidente : supprimer le DataFrame"
```

### cProfile : le profileur standard Python

`cProfile` instrumente chaque appel de fonction et mesure le temps cumulé. Il fonctionne "autour" du code sans le modifier :

```python
import cProfile, pstats, io

profiler = cProfile.Profile()
profiler.enable()
for _ in range(100):           # 100 appels pour avoir des statistiques stables
    predict_lgbm_full(sample_features)
profiler.disable()

stream = io.StringIO()
pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
```

**`sort_stats("cumulative")`** : trie par temps cumulé (inclut les sous-appels). C'est la métrique la plus utile pour trouver le goulot de bout en bout.

**Pourquoi 100 appels et pas 1 ?** Pour lisser le bruit (cache CPU, GC Python, etc.) et obtenir des moyennes fiables.

### timeit : mesurer des snippets précis

Une fois le goulot identifié avec cProfile, `timeit` mesure chaque étape séparément avec une précision sous-milliseconde :

```python
import timeit

N = 500

# Étape A : construction du DataFrame
t_df = timeit.timeit(
    lambda: pd.DataFrame([client_data]).reindex(columns=feature_names, fill_value=0),
    number=N,
) / N * 1000  # → temps moyen en ms

# Étape B : predict_proba seul (DataFrame pré-construit)
t_proba = timeit.timeit(
    lambda: model.predict_proba(df_sample)[:, 1][0],
    number=N,
) / N * 1000
```

**`number=N`** : le code est exécuté N fois, le temps total est divisé par N → moyenne.

### Ce que le profiling a révélé dans ce projet

```
┌──────────────────────────────────────────────────────────────┐
│  Décomposition de la latence LightGBM (500 runs)             │
│                                                              │
│  DataFrame build + reindex  ████████████████  0.52 ms  73%  │
│  predict_proba              ████             0.16 ms  23%   │
│  Dict serialization         █                0.03 ms   4%   │
│                                                              │
│  → Le bottleneck n'est PAS le modèle. C'est Pandas.         │
└──────────────────────────────────────────────────────────────┘
```

**Pourquoi `pd.DataFrame([dict]).reindex(...)` est-il lent ?**

1. **`pd.DataFrame([dict])`** : Python construit un DataFrame depuis un dict → allocations mémoire, inférence de dtypes, copie
2. **`.reindex(columns=feature_names, fill_value=0)`** : réordonne les 419 colonnes dans le bon ordre → scan de toutes les colonnes, remplissage des NaN
3. Ces deux opérations font plusieurs milliers d'opérations Python pour un seul client

**La solution** : passer directement en numpy array — zéro pandas, zéro réordonnancement, juste une allocation simple.

---

## 3. ONNX : le format universel des modèles ML

### L'analogie du PDF

Un document Word `.docx` ne s'ouvre pas forcément bien sur tous les ordinateurs, et vous ne pouvez pas l'ouvrir dans LibreOffice sans conversions. Un PDF `.pdf` s'ouvre partout, identiquement, sur n'importe quel lecteur.

ONNX (Open Neural Network Exchange) est le **PDF des modèles ML** : un format standardisé qui peut être exécuté par n'importe quel runtime compatible, quelle que soit la bibliothèque d'entraînement.

```
                    ┌──────────────────────────────────┐
  Entraînement      │  scikit-learn, LightGBM, PyTorch │
  (votre langage)   │  XGBoost, TensorFlow, Keras...   │
                    └──────────────┬───────────────────┘
                                   │  conversion
                                   ▼
                    ┌──────────────────────────────────┐
  Format universel  │         model.onnx               │
                    │  Graphe de calcul standardisé    │
                    │  (opérations ONNX officielles)   │
                    └──────────────┬───────────────────┘
                                   │  exécution
                                   ▼
                    ┌──────────────────────────────────┐
  Runtime           │  ONNX Runtime, TensorRT, CoreML, │
  (n'importe lequel)│  OpenVINO, DirectML...           │
                    └──────────────────────────────────┘
```

### Pourquoi ONNX est plus rapide que l'original

Un modèle LightGBM en Python transporte avec lui :
- Le runtime Python (GIL, gestion mémoire dynamique)
- Le framework LightGBM (couche scikit-learn sklearn wrapper)
- La validation Pandas des inputs

ONNX Runtime, lui, exécute un **graphe de calcul compilé** :
- Code natif (C++) directement, sans Python interprété
- Graphe optimisé à la compilation (fusion d'opérations, élimination de code mort)
- Pas de validation de type runtime — le format ONNX garantit les types à la conversion

Pour un modèle de type "forêt de décision" (comme LightGBM), ONNX Runtime implémente une traversée d'arbre ultra-optimisée en C++ avec vectorisation SIMD, bien plus rapide que la version Python.

### Le graphe ONNX d'un arbre de décision

```
Input: float_input [1, 419]
         │
         ▼
  TreeEnsembleClassifier
  (opérateur ONNX officiel)
  ┌──────────────────────────────────────────┐
  │  Arbre 1 : if feat[42] < 0.5 → nœud A  │
  │  Arbre 2 : if feat[7] < 1.2  → nœud B  │
  │  ...  (des centaines d'arbres)           │
  │  Combinaison : softmax des scores        │
  └──────────────────────────────────────────┘
         │
         ├──→ Output 0 : label [1]         (int64 : classe prédite)
         └──→ Output 1 : probabilities     (ZipMap : {classe → proba})
```

Tout le "boosting" LightGBM est encodé dans **un seul opérateur ONNX** (`TreeEnsembleClassifier`), qui est exécuté en C++ compilé.

---

## 4. La conversion LightGBM → ONNX avec onnxmltools

### Pourquoi onnxmltools et pas skl2onnx ?

`skl2onnx` est le convertisseur officiel pour scikit-learn. Mais LightGBM n'est pas nativement sklearn — c'est une bibliothèque externe qui fournit un **wrapper sklearn** (`LGBMClassifier`). Ce wrapper n'a pas de convertisseur enregistré dans skl2onnx.

`onnxmltools` est une bibliothèque compagnon qui ajoute des convertisseurs pour les bibliothèques non-sklearn : **LightGBM, XGBoost, H2O, Spark ML**. Elle utilise les mêmes primitives ONNX mais connaît l'architecture interne de ces frameworks.

```
skl2onnx seul :
  LGBMClassifier → ??? → MissingShapeCalculator ERROR

onnxmltools :
  LGBMClassifier.booster_ → TreeEnsembleClassifier (ONNX) → model.onnx ✓
```

**Attention** : onnxmltools a sa propre classe `FloatTensorType` — différente de celle de skl2onnx. Mélanger les deux cause une erreur de type à la conversion :

```python
# ❌ FAUX : skl2onnx.FloatTensorType avec onnxmltools.convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
convert_lightgbm(booster, initial_types=[("float_input", FloatTensorType([None, 419]))])
# → RuntimeError: wrong type <class 'skl2onnx.common.data_types.FloatTensorType'>

# ✓ CORRECT : onnxmltools.FloatTensorType avec onnxmltools.convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
convert_lightgbm(booster, initial_types=[("float_input", FloatTensorType([None, 419]))])
```

### Le code de conversion (notebook 06)

```python
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

# On passe le Booster sous-jacent (pas le wrapper sklearn)
booster = model_lgbm.booster_

initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
#                 ↑ nom du tenseur        ↑ shape : batch illimité, 419 features

onnx_model = convert_lightgbm(
    booster,
    initial_types=initial_type,
    target_opset=8,   # version ONNX (8 = compatible avec toutes les plateformes)
)

with open("artifacts/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**`target_opset=8`** : version du standard ONNX à utiliser. Une version plus récente peut apporter des optimisations, mais une version plus ancienne garantit une meilleure compatibilité cross-runtime. L'opset 8 couvre `TreeEnsembleClassifier` parfaitement.

**`booster_` vs `LGBMClassifier`** : le booster contient les arbres. Le wrapper sklearn ajoute des couches Python (validation, interface fit/predict) qui n'ont pas besoin d'être dans l'ONNX.

### Ce que onnxmltools fait en interne

```
booster.dump_model()          → représentation JSON des arbres
         │
         ▼
  Lecture des split conditions (feature index, threshold, direction)
  + Lecture des leaf values
  + Lecture de la structure d'ensemble (nombre d'arbres, classes)
         │
         ▼
  Construction du graphe ONNX :
  TreeEnsembleClassifier {
      nodes_featureids: [42, 7, 185, ...]    ← features utilisées
      nodes_values: [0.5, 1.2, -0.3, ...]    ← seuils des splits
      nodes_modes: [BRANCH_LEQ, ...]          ← type de comparaison
      leaf_targetids: [0, 1, 0, 1, ...]       ← classe de chaque feuille
      leaf_weights: [0.12, 0.87, ...]         ← log-odds des feuilles
  }
         │
         ▼
  Sérialisation protobuf → model.onnx
```

---

## 5. ONNX Runtime : exécuter le modèle sans framework ML

### Chargement et configuration

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1   # 1 thread par opération (Render free tier = CPU partagé)
opts.inter_op_num_threads = 1   # 1 thread entre opérations

session = ort.InferenceSession(
    "artifacts/model.onnx",
    opts,
    providers=["CPUExecutionProvider"],  # pas de GPU dans ce projet
)
```

**`intra_op_num_threads = 1`** : pourquoi limiter à 1 thread sur un free tier cloud ?
- Le free tier Render partage le CPU entre plusieurs instances
- Utiliser 4 threads sur un CPU partagé à 2 cœurs crée de la contention et *ralentit* l'exécution
- Pour la single-row inference, le parallélisme intra-op n'apporte rien (un seul client)

**`CPUExecutionProvider`** : ONNX Runtime supporte CUDA, TensorRT, CoreML, DirectML. On force CPU pour garantir le comportement sur Render.

### Le format d'entrée/sortie

```python
# Inspecter le modèle
session.get_inputs()[0].name   # "float_input"
session.get_inputs()[0].shape  # [None, 419]  (None = batch dynamique)
session.get_inputs()[0].type   # "tensor(float)"

session.get_outputs()[0].name  # "label"          ← classe prédite (int64)
session.get_outputs()[0].type  # "tensor(int64)"
session.get_outputs()[1].name  # "probabilities"   ← probabilités par classe
session.get_outputs()[1].type  # "seq(map(int64,tensor(float)))"
```

**`seq(map(int64,tensor(float)))`** : ce type ONNX représente une **séquence de dictionnaires** `{int64 → float}`. En Python, cela se matérialise comme :

```python
results = session.run(None, {"float_input": X_np})
# results[1] = [{0: 0.607, 1: 0.393}]   ← liste de 1 dict (batch=1)
# results[1][0] = {0: 0.607, 1: 0.393}  ← dict pour le 1er client
# results[1][0][1] = 0.393               ← probabilité de la classe 1 (défaut)
```

C'est le format **ZipMap** (hérité d'ONNX ML) — plus lisible que les arrays bruts, et l'indexage `[1]` sur un dict Python fonctionne identiquement à l'indexage `[1]` sur un tableau numpy : les deux retournent la probabilité de la classe 1.

---

## 6. Le dual-path dans l'API : ONNX d'abord, LightGBM en fallback

### Le principe

L'API ne *remplace pas* LightGBM par ONNX — elle les garde **tous les deux**. Au démarrage, elle cherche `model.onnx`. S'il existe, elle l'utilise. Sinon, elle continue avec LightGBM. Zéro breaking change, zéro configuration manuelle.

```
API startup (api/predict.py)
│
├── Charger model.pkl → self.model (LGBMClassifier)   ← toujours fait
│
└── Chercher artifacts/model.onnx
    ├── Trouvé → charger ONNX Runtime session
    │            self._use_onnx = True
    │            log: {"event": "onnx_loaded"}
    │
    └── Absent → self._use_onnx = False
                 log: warning (si chargement échoue)
                 → fallback silencieux vers LightGBM

Appel /predict :
    self._use_onnx → _predict_onnx()
    sinon          → _predict_lgbm()
```

### Les deux chemins d'inférence côte à côte

```python
def _predict_lgbm(self, client_data: dict) -> float:
    # ① Créer un DataFrame depuis le dict
    df = pd.DataFrame([client_data])
    # ② Réordonner les 419 colonnes dans le bon ordre
    df = df.reindex(columns=self.feature_names, fill_value=0)
    # ③ Appeler predict_proba via le wrapper sklearn
    return float(self.model.predict_proba(df)[:, 1][0])
    # Total : ~0.71 ms (dont 73% pour ① et ②)


def _predict_onnx(self, client_data: dict) -> float:
    # ① Allouer un tableau numpy de zéros
    X = np.zeros((1, len(self.feature_names)), dtype=np.float32)
    # ② Remplir uniquement les features présentes (sparse lookup)
    for name, value in client_data.items():
        try:
            X[0, self.feature_names.index(name)] = float(value)
        except ValueError:
            pass  # feature inconnue → ignorée silencieusement
    # ③ Inférence C++ via ONNX Runtime
    outputs = self._ort_session.run(
        [self._ort_output_name],        # "probabilities"
        {self._ort_input_name: X}       # "float_input"
    )
    return float(outputs[0][0][1])
    # Total : ~0.01 ms
```

**Pourquoi `np.zeros` et pas `np.empty` ?** `zeros` garantit que les features absentes valent 0 (comme `fill_value=0` du reindex LightGBM). `empty` laisserait des valeurs arbitraires en mémoire → résultats incorrects.

**Pourquoi `dtype=np.float32` et pas `float64` ?** ONNX Runtime a été configuré pour recevoir `tensor(float)` = float32. Envoyer du float64 lèverait une erreur. LightGBM Python, lui, accepte les deux (conversion interne).

### `feature_names.index(name)` : la recherche linéaire

Pour chaque feature de la requête, on fait un `.index()` sur une liste de 419 éléments. C'est une recherche linéaire O(n). Pour 10 features non-nulles, c'est 10 × 419 = 4 190 comparaisons maximum.

En pratique, c'est négligeable (< 0.001 ms) car :
- Les features de la requête sont peu nombreuses (client sparse)
- La liste est petite (419 éléments)
- Python liste.index() est implémenté en C

Si les performances devenaient critiques (10k req/s), on pourrait précalculer `{name: index}` en dict pour des lookups O(1). Ce n'est pas nécessaire ici.

### Le `self.model` reste `LGBMClassifier`

Point important pour la compatibilité : même quand ONNX est actif, `self.model` contient toujours le `LGBMClassifier` d'origine. L'endpoint `/model-info` expose `model.__class__.__name__` = `"LGBMClassifier"` — cela reste vrai, car on décrit le **modèle entraîné**, pas le runtime d'exécution.

```python
# test_api.py — ce test continue de passer sans modification
def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.json()["model_type"] == "LGBMClassifier"  # ✓ toujours vrai
```

---

## 7. Validation numérique : vérifier que les deux modèles sont équivalents

### Pourquoi cette étape est-elle critique ?

La conversion ONNX n'est pas parfaite — elle réordonne les opérations, change les types flottants (float64 → float32), et utilise une implémentation différente de la traversée d'arbre. Un bug de conversion silencieux pourrait faire prédire des valeurs légèrement (ou très) différentes.

La validation numérique quantifie **l'écart accepté** :

```python
proba_lgbm = model.predict_proba(df_sample)[:, 1][0]   # ex: 0.39234817
proba_onnx = session.run(...)[0][0][1]                   # ex: 0.39234810

delta = abs(proba_lgbm - proba_onnx)   # 7.31e-08
assert delta < 1e-4, f"|Δ| = {delta}"
```

**Pourquoi 1e-4 et pas 1e-10 ?**
- La précision float32 est ~7 décimales significatives
- La traversée d'arbre implique des accumulations de valeurs float32
- Un Δ de 1e-4 = 0.0001 est imperceptible sur une probabilité [0, 1]
- Sur un seuil à 0.494, un Δ de 1e-4 ne changerait jamais la décision

**Résultat mesuré** : Δ = 7.31e-08 (bien en dessous de 1e-4 ✓)

### D'où vient la différence résiduelle ?

```
LightGBM (float64)              ONNX Runtime (float32)
────────────────────────────    ────────────────────────────
Accumule les leaf_values        Accumule les mêmes valeurs
en double précision (64 bits)   en simple précision (32 bits)
→ 15-17 décimales signif.       → 6-7 décimales signif.
                                → arrondi différent au 7e chiffre
```

C'est une perte de précision **attendue et documentée** de la conversion float64 → float32, sans impact métier.

---

## 8. Benchmark : mesurer le gain réel

### Méthodologie

```python
import time

N_BENCH = 500

# Warm-up : les premières inférences sont plus lentes (JIT, cache CPU)
for _ in range(10):
    benchmark_lgbm()
    benchmark_onnx()

# Mesure après warm-up
times_lgbm = []
for _ in range(N_BENCH):
    t0 = time.perf_counter()    # horloge haute résolution (nanoseconde)
    benchmark_lgbm()
    times_lgbm.append((time.perf_counter() - t0) * 1000)  # en ms
```

**`time.perf_counter()`** : l'horloge la plus précise disponible en Python — typiquement à la nanoseconde, contrairement à `time.time()` qui peut avoir une résolution de 15ms sur Windows.

**`timeit.timeit`** (utilisé pour le profiling par étapes) vs **boucle `perf_counter`** (utilisé pour le benchmark final) :
- `timeit` minimise les effets des GC et du système, mais ne donne que la moyenne
- La boucle `perf_counter` donne la **distribution complète** (P50, P95, std) — nécessaire pour la prod

### Les percentiles P50 et P95

```
Distribution de latence sur 500 runs (ONNX)
│
│ 400 ┤                  ████████
│ 300 ┤              ████████████
│ 200 ┤          ████████████████
│ 100 ┤       ███████████████████
│     └──┬──────────┬──────────┬────
│       0.008     0.010      0.015  (ms)
│          ↑         ↑
│         P50      P95
```

- **P50 (médiane)** : 50% des requêtes sont plus rapides. Représente l'expérience "normale".
- **P95** : 95% des requêtes sont plus rapides. Les 5% restants sont des "outliers" (GC Python, contention CPU). C'est la métrique SLA standard : "95% des requêtes répondent en moins de X ms".
- **Mean** : peut être biaisée par quelques valeurs extrêmes — moins fiable que la médiane.

### Résultats mesurés

```
┌──────────────────────────────────────────────────────────────┐
│  Benchmark final — 500 runs, single-row, Mac M-series        │
├────────────────┬───────────┬──────────┬──────────┬───────────┤
│ Engine         │ Mean (ms) │ P50 (ms) │ P95 (ms) │ Artefact  │
├────────────────┼───────────┼──────────┼──────────┼───────────┤
│ LightGBM       │   0.71    │   0.70   │   0.77   │  1.12 MB  │
│ ONNX Runtime   │   0.01    │   0.01   │   0.01   │  0.71 MB  │
├────────────────┼───────────┼──────────┼──────────┼───────────┤
│ Speedup        │  57.7×    │  58.1×   │  60.9×   │  −37%     │
└────────────────┴───────────┴──────────┴──────────┴───────────┘
```

**Interprétation du 57× :**
- Sur un seul cœur, ONNX exécute la traversée d'arbres en ~10 µs (microseconde)
- LightGBM passe ~520 µs en overhead Pandas + ~160 µs en predict_proba
- L'optimisation ONNX élimine les 520 µs Pandas *et* accélère les 160 µs predict_proba

**Note sur la généralisabilité :**
- Ces mesures sont sur CPU M-series (Apple Silicon) en développement local
- Sur Render free tier (CPU Intel partagé), les valeurs absolues seront différentes (plus lentes)
- Le **ratio** de speedup sera similaire, car les deux chemins s'exécutent sur le même matériel

### Taille des artefacts

```
model.pkl  = 1.12 MB  ← pickle Python : inclut les métadonnées sklearn, les paramètres...
model.onnx = 0.71 MB  ← protobuf : seulement les arbres, format binaire compact

Ratio : 0.71 / 1.12 = 0.63 → l'ONNX est 37% plus petit
```

Pourquoi ONNX est plus compact ? Le pkl contient des objets Python complets (avec leurs méthodes, attributs, etc.). L'ONNX ne contient que les données nécessaires à l'inférence (split conditions et leaf values), sérialisées en protobuf binaire.

---

## 9. Vue d'ensemble et résultats

### Le flux complet d'optimisation

```
notebooks/06_inference_optimization.ipynb
│
├── Section 0 : Chargement model.pkl, feature_names.json
│
├── Section 1 : cProfile (100 runs LightGBM)
│   → Identifie pd.DataFrame comme goulot principal
│
├── Section 2 : timeit granulaire
│   → Quantifie : DataFrame 73%, predict_proba 23%, dict 4%
│
├── Section 3 : Conversion ONNX
│   model_lgbm.booster_ → onnxmltools → artifacts/model.onnx
│
├── Section 4 : Validation numérique
│   |LGBMproba - ONNXproba| = 7.31e-08 < 1e-4 ✓
│
├── Section 5 : Benchmark 500 runs
│   ONNX 57× plus rapide (P50: 0.01ms vs 0.70ms)
│
├── Section 6 : Mémoire (tracemalloc)
│   model.onnx 37% plus petit que model.pkl
│
├── Section 7 : Graphiques → charts_eda/06_inference_benchmark.png
│
└── Section 8 : Synthèse → artifacts/inference_benchmark.json
```

### Intégration dans le pipeline de déploiement

```
git push → GitHub Actions
│
├── pytest tests/ → 19 tests passent (dont test_model_info "LGBMClassifier" ✓)
│
├── docker build
│   └── COPY artifacts/model.onnx .   ← inclus automatiquement
│       (model.onnx est dans .dockerignore ? Non → copié)
│
└── deploy Render
    └── API startup log :
        {"event": "onnx_loaded", "path": "artifacts/model.onnx"}
        → _use_onnx = True → toutes les requêtes /predict utilisent ONNX
```

### Tableau de synthèse des concepts

| Concept | Rôle | Outil | Fichier |
|---|---|---|---|
| **Profiling** | Trouver le vrai goulot | `cProfile`, `timeit` | notebook 06, section 1-2 |
| **Format ONNX** | Standard de sérialisation des modèles ML | protobuf + opset | `artifacts/model.onnx` |
| **Conversion** | LightGBM → ONNX | `onnxmltools` | notebook 06, section 3 |
| **Runtime** | Exécution native C++ du graphe ONNX | `onnxruntime` | `api/predict.py` |
| **Dual-path** | ONNX si dispo, LightGBM en fallback | `_use_onnx` flag | `api/predict.py` |
| **Validation** | Vérifier l'équivalence numérique | assert |Δ| < 1e-4 | notebook 06, section 4 |
| **Benchmark** | Mesurer le gain (P50, P95) | `time.perf_counter` | notebook 06, section 5 |

### Décisions de design et leurs raisons

| Décision | Raison |
|---|---|
| Garder `self.model` LGBMClassifier | `/model-info` continue de retourner "LGBMClassifier" — aucun test à modifier |
| Fallback silencieux si ONNX absent | CI passe même si `model.onnx` n'est pas dans le repo |
| `intra_op_num_threads=1` | Free tier Render = CPU partagé, multi-thread crée de la contention |
| `float32` et non `float64` | Format attendu par le modèle ONNX (converti en float32 à la compilation) |
| `np.zeros` et non `np.empty` | Garantit que les features absentes = 0, identique à `fill_value=0` de reindex |
| `onnxmltools` et non `skl2onnx` | skl2onnx n'a pas de convertisseur pour LGBMClassifier |
| `booster_` et non `LGBMClassifier` | onnxmltools attend le Booster interne, pas le wrapper sklearn |

---

> **Référence** : `notebooks/06_inference_optimization.ipynb` — le notebook exécutable qui génère `model.onnx`, effectue la validation et produit tous les graphiques de benchmark.
