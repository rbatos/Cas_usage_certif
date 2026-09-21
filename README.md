# Cas d'usage — Orientation et tri multimodal des demandeurs d'emploi

Créer un système d’IA pour prédire le délai de retour à l’emploi (3 classes : `<6 mois`, `6-12 mois`, `>12 mois`) à partir de données multimodales (tabulaires + texte), en respectant l’éthique (RGPD) et les contraintes métiers.

---

## ✅ État d’avancement du projet

### Compréhension du Sujet et Préparation des Données
- [x] Compréhension du sujet (analyse + livrables)
- [x] EDA (chargement, variables sensibles, feature engineering, encodage)
- [x] Préparation des scénarios

### Modélisation et Évaluation
- [x] Choix des modèles à comparer (Random Forest, LightGBM, XGBoost, ...)
- [x] Définition des métriques d’évaluation
- [x] Entraînement et optimisation
- [x] Analyse des résultats

### Analyse Éthique et Réglementaire
- [x] Conformité RGPD / CNIL
- [x] Analyse biais / discrimination
- [?] Responsabilité juridique

### Industrialisation et Déploiement
- [x] Architecture cible
- [x] Contraintes techniques
- [x] CI/CD et monitoring

### Réflexion sur les Erreurs Critiques et Optimisation
- [x] Identification des erreurs critiques
- [ ] Stratégies de réduction
- [ ] Évaluation des améliorations

### Rédaction du Rapport et Préparation de la Soutenance
- [ ] Rédaction du rapport
- [ ] Préparation du support de soutenance

---

## 📁 Structure actuelle

```text
Cas_usage_certif/
├── flux_donnees.md
├── pytest.ini
├── README.md
├── requirements.txt
├── Dockerfile                                => image API et MLflow (uvicorn + modèle)
├── docker-compose.yml                        => orchestration locale api + ui + mlflow (build)
├── docker-compose.prod.yml                   => orchestration cible (images GHCR, déploiement CI/CD)
├── ruff.toml                                 => configuration du linter (job `lint`)
├── .github/
│   └── workflows/
│       └── ci.yml                            => CI/CD : lint, tests, entraînement, build & push GHCR, déploiement
├── app
│   ├── main.py
│   ├── middleware.py
│   ├── requirements.txt                      => dépendances minimales de l'image API
│   └── schemas.py
├── data/
│   └── dataset_trajectoire_emploi_Sujet Examen CISIA - Promo U...     => .gitignore pour le moment... à réfléchir!
├── docs/
│   └── Sujet Examen CISIA.md
├── logs/
│   └── api.log                              => logs middleware
├── mlruns/                                   => tracking MLflow local partagé avec Docker
├── models/
│   ├── modele_lgbm_v1.0_S1_multimodale_complete.joblib
│   ├── *_metadata.json                       => versions, métriques et hyperparamètres
│   ├── registre_modeles_sauvegardes.csv      => registre local des artefacts
├── notebooks/
│   ├── journal-de-bord.ipynb
│   └── matrice-notebook-romain.ipynb
├── services/
│   └── ui-streamlit/                         => interface web conseiller
│      ├── app.py                             => saisie features -> prédiction + probabilités
│      ├── Dockerfile
│      └── requirements.txt
└── tests
    └── test_api.py
```

---

## 🚀 Exécution (Windows / VS Code)

1. Créer et activer l’environnement :
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Lancer Jupyter :
```powershell
jupyter notebook notebooks/matrice-notebook-romain.ipynb
```

3. Test vi uvicorn :
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```
=> URL utilisable : `http://127.0.0.1:8000/docs`

4. Lancer l'interface conseiller (Streamlit) — l'API doit tourner en parallèle :
```powershell
$env:API_URL="http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run services\ui-streamlit\app.py
```
=> URL utilisable : `http://localhost:8502`

5. Tests Pytest
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

6. Suivi des entraînements avec MLflow
```powershell
New-Item -ItemType Directory -Force mlruns
\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri .\mlruns --port 5000
```
=> Interface de suivi : `http://127.0.0.1:5000`

Chaque appel à `/train` crée un run dans l’expérience `orientation-retour-emploi`.
Les hyperparamètres LightGBM, les métriques de validation, la baseline, le nombre de
lignes et le nombre de feedbacks sont enregistrés. Les entraînements rejetés sont
conservés dans MLflow mais ne créent pas de version de modèle.

Lorsqu’un modèle est accepté par le garde-fou `f1_macro`, il est enregistré dans le
Model Registry sous `orientation-retour-emploi` et l’alias contrôlé `champion` est
déplacé vers cette nouvelle version. Les paramètres peuvent être configurés avec
`MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_MODEL_NAME` et
`MLFLOW_MODEL_ALIAS`.

---

## 🐳 Exécution avec Docker

L'API, l'UI Streamlit et l'interface MLflow peuvent tourner en conteneurs, orchestrés par `docker-compose.yml` :

```powershell
docker compose up --build
```

- API : `http://localhost:8000` (docs sur `/docs`)
- Interface conseiller : `http://localhost:8501`
- Interface MLflow : `http://127.0.0.1:5000`

Le service `api` monte `models/`, `data/`, `logs/` et `mlruns/` en volumes. Le service `mlflow` réutilise l'image API et monte le même dossier `mlruns/` afin d'afficher les runs produits par l'API. Le lifespan de l'API charge le modèle au démarrage et applique une politique **fail-fast** : si l'artefact modèle est absent ou corrompu, le conteneur `api` s'arrête en erreur (`docker compose ps` affiche `unhealthy`/exit) plutôt que de répondre avec un modèle cassé. Le service `ui` attend que `api` soit `healthy` (`depends_on: condition: service_healthy`) avant de démarrer, et le joint via `API_URL=http://api:8000`.

Pour arrêter et supprimer les conteneurs :
```powershell
docker compose down
```

---

## ⚙️ CI/CD (GitHub Actions)

Le workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) automatise l'ensemble du cycle de vie :

1. **`lint`** (à chaque push et pull request) : vérifie le style et la qualité du code avec `ruff check` et `ruff format --check` (config dans [`ruff.toml`](ruff.toml)). Bloque la suite si le code n'est pas conforme.
2. **`test`** (après `lint`) : installe `app/requirements.txt` + `pytest`/`httpx`, puis exécute les tests unitaires de l'API (hors entraînement).
3. **`train-validation`** (après `lint`, en parallèle de `test`) : exécute spécifiquement les tests de la route `/train` (entraînement, promotion vs baseline `f1_macro`, tolérance de dégradation) pour garantir la reproductibilité du pipeline d'entraînement avant toute mise en production.
4. **`build-and-push`** (uniquement sur push vers `main` ou tag `v*`, après succès de `test` et `train-validation`) : construit les images `api` et `ui` en parallèle (matrice) et les publie sur GitHub Container Registry (GHCR), taguées avec le SHA du commit, `latest` et la branche/tag.
5. **`deploy`** (uniquement sur push vers `main`, après `build-and-push`) : s'exécute sur un **runner self-hosted** installé sur la machine cible, s'authentifie à GHCR puis redéploie la stack via [`docker-compose.prod.yml`](docker-compose.prod.yml) (`docker compose pull` + `up -d`), garantissant un déploiement reproductible et traçable (image identifiée par son SHA de commit).

Déclencheurs : `push` sur `main`, tags `v*`, `pull_request` vers `main`, et déclenchement manuel (`workflow_dispatch`).

### Déploiement local via runner self-hosted (WSL/PC)

Le job `deploy` tourne sur un **runner self-hosted** (`runs-on: self-hosted`) installé directement sur la machine cible (ici : WSL), plutôt que via SSH depuis les runners cloud GitHub — ceux-ci ne peuvent pas atteindre une machine derrière une box/NAT sans tunnel.

Installation du runner (dans WSL, une seule fois) :

1. GitHub → dépôt → **Settings → Actions → Runners → New self-hosted runner**, choisir *Linux*.
2. Suivre les commandes affichées (téléchargement de l'archive, `./config.sh --url ... --token ...`).
3. Installer le runner comme service persistant pour qu'il tourne en arrière-plan :
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```
4. Vérifier qu'il apparaît **Idle** dans *Settings → Actions → Runners*.

Le runner doit avoir Docker installé et l'utilisateur qui l'exécute doit appartenir au groupe `docker` (`sudo usermod -aG docker $USER`, puis relancer une session).

Aucun secret `DEPLOY_*` n'est nécessaire avec cette approche : le job s'exécute directement sur la machine cible et n'a besoin que de `GITHUB_TOKEN` (fourni automatiquement) pour s'authentifier à GHCR.

---

## 🧠 Méthodologie retenue (cible)

- Problème de **classification multiclasse** (`<6 mois`, `6-12 mois`, `>12 mois`)
- Approche **multimodale** : fusion variables tabulaires + texte
- Comparaison équitable des modèles (mêmes splits, mêmes métriques)
- Intégration des contraintes **éthiques et réglementaires** dès la conception

---

## 🧪 Données / Features (à préciser dans le notebook)

- Variables tabulaires socio-pro
- Variables textuelles (commentaires)
- Variable cible : délai de retour à l’emploi (3 classes)

---

## 📌 Livrables attendus

- Notebook d’analyse (`notebooks/matrice-notebook-romain.ipynb`)
- Journal de bord (`notebooks/journal-de-bord.ipynb`)
- Support de soutenance
