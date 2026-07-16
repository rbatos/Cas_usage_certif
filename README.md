# Cas d'usage — Orientation et tri multimodal des demandeurs d'emploi

Créer un système d’IA pour prédire le délai de retour à l’emploi (3 classes : `<6 mois`, `6-12 mois`, `>12 mois`) à partir de données multimodales (tabulaires + texte), en respectant l’éthique (RGPD) et les contraintes métiers.

---

## ✅ État d’avancement du projet

### Compréhension du Sujet et Préparation des Données
- [ ] Compréhension du sujet (analyse + livrables)
- [ ] EDA (chargement, variables sensibles, feature engineering, encodage)
- [ ] Préparation des scénarios

### Modélisation et Évaluation
- [ ] Choix des modèles à comparer
- [ ] Définition des métriques d’évaluation
- [ ] Entraînement et optimisation
- [ ] Analyse des résultats

### Analyse Éthique et Réglementaire
- [ ] Conformité RGPD / CNIL
- [ ] Analyse biais / discrimination
- [ ] Responsabilité juridique

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
├── flux_donnees.md
├── README.md
├── requirements.txt
├── data/
│   └── dataset_trajectoire_emploi_Sujet Examen CISIA - Promo U...
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
jupyter notebook notebooks/TODO.ipynb
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
- Variables textuelles (freins, parcours, commentaires)
- Variable cible : délai de retour à l’emploi (3 classes)

---

## 📌 Livrables attendus

- Notebook d’analyse (`notebooks/`)
- Journal de bord (`notebooks/journal-de-bord.ipynb`)
- Support de soutenance
