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
- [ ] Architecture cible
- [ ] Contraintes techniques
- [ ] CI/CD et monitoring

### Réflexion sur les Erreurs Critiques et Optimisation
- [ ] Identification des erreurs critiques
- [ ] Stratégies de réduction
- [ ] Évaluation des améliorations

### Rédaction du Rapport et Préparation de la Soutenance
- [ ] Rédaction du rapport
- [ ] Préparation du support de soutenance

---

## 📁 Structure actuelle

```text
Cas_usage_certif/
├── README.md
├── requirements.txt
├── app
│   ├── main.py
│   └── schemas.py
├── data/
│   └── dataset_trajectoire_emploi_Sujet Examen CISIA - Promo U...     => .gitignore pour le moment... à réfléchir!
├── docs/
│   └── Sujet Examen CISIA.md
├── modele/
│   ├── modele_final.joblib                  => modèle entraîné sauvegardé
│   ├── metadonnees_modele.json              => version, métriques, features, mapping cible
│   ├── registre_modeles_sauvegardes.csv     => liste des modèles sauvegardés
└── notebooks/
    ├── journal-de-bord.ipynb
    └── matrice-notebook-romain.ipynb
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
