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
├── Dockerfile                                => image API (uvicorn + modèle)
├── docker-compose.yml                        => orchestration api + ui
├── .github/
│   └── workflows/
│       └── ci.yml                            => CI/CD : tests, build & push images GHCR
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
├── modele/
│   ├── modele_final.joblib                  => modèle entraîné sauvegardé
│   ├── metadonnees_modele.json              => version, métriques, features, mapping cible
│   ├── registre_modeles_sauvegardes.csv     => liste des modèles sauvegardés
└── notebooks/
│   ├── journal-de-bord.ipynb
│   └── matrice-notebook-romain.ipynb
├── services/
│   └── ui-streamlit/                     => interface web conseiller (saisie features -> prédiction + probas)
|      ├── app.py
|      ├── Dockerfile
|      └── requirements.txt
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

---

## 🐳 Exécution avec Docker

L'API et l'UI Streamlit peuvent aussi tourner en conteneurs, orchestrés par `docker-compose.yml` :

```powershell
docker compose up --build
```

- API : `http://localhost:8000` (docs sur `/docs`)
- Interface conseiller : `http://localhost:8501`

Le service `api` monte `models/`, `data/` et `logs/` en volumes. Le lifespan de l'API charge le modèle au démarrage et applique une politique **fail-fast** : si l'artefact modèle est absent ou corrompu, le conteneur `api` s'arrête en erreur (`docker compose ps` affiche `unhealthy`/exit) plutôt que de répondre avec un modèle cassé. Le service `ui` attend que `api` soit `healthy` (`depends_on: condition: service_healthy`) avant de démarrer, et le joint via `API_URL=http://api:8000`.

Pour arrêter et supprimer les conteneurs :
```powershell
docker compose down
```

---

## ⚙️ CI/CD (GitHub Actions)

Le workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) automatise deux étapes :

1. **`test`** (à chaque push et pull request) : installe les dépendances de `app/requirements.txt` + `pytest`/`httpx`, puis exécute `pytest -v`. Un test qui échoue bloque tout le reste.
2. **`build-and-push`** (uniquement sur push vers `main` ou tag `v*`, après succès de `test`) : construit les images `api` et `ui` en parallèle (matrice) et les publie sur GitHub Container Registry (GHCR), taguées avec le SHA du commit, `latest` et la branche/tag.

Déclencheurs : `push` sur `main`, tags `v*`, `pull_request` vers `main`, et déclenchement manuel (`workflow_dispatch`).

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
