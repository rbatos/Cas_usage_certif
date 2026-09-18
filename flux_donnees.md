# Schéma des flux de données

## Schéma

```mermaid
flowchart LR
    Client[Client HTTP] --> API[API FastAPI]
    API --> Route{Route appelée}

    Route -->|GET /health| Health[Contrôle de santé]
    Health --> HealthModel{Modèle chargé ?}
    HealthModel -->|Oui| HealthOK[Réponse 200 : status ok]
    HealthModel -->|Non| HealthKO[Réponse 200 : status degraded]

    Route -->|POST /predict| Validation[Validation Pydantic]
    Validation -->|Payload invalide| Error422[Réponse 422]
    Validation -->|Payload valide| ModelCheck{Modèle disponible ?}
    ModelCheck -->|Non| Error503[Réponse 503 : modèle indisponible]
    ModelCheck -->|Oui| Preparation[Préparation des features]
    Preparation --> Features[Extraction du département<br/>depuis le code INSEE]
    Features --> Pipeline[Pipeline LightGBM]
    Pipeline --> Prediction[Classe prédite<br/>+ probabilités]
    Prediction --> Response200[Réponse 200]

    Route -->|POST /train| Dataset[Lecture du fichier CSV]
    Dataset --> Training[Réentraînement du pipeline]
    Training --> Save[Sauvegarde du modèle]
    Save --> TrainOK[Réponse 200 : modèle entraîné]

    API -.-> Logs[(logs/api.log)]
    Route -.-> Logs

    classDef ok fill:#d1fae5,stroke:#16a34a,stroke-width:2px,color:#166534
    classDef error fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b
    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e

    class HealthOK,Response200,TrainOK ok
    class Error422,Error503 error
    class API,Preparation,Features,Pipeline,Prediction,Dataset,Training,Save process
    class Route,HealthModel,ModelCheck decision
```

## Légende

Le client HTTP envoie une requête à l'API FastAPI, qui la dirige vers la route demandée. Pour `/predict`, le payload est validé, puis le modèle LightGBM produit une classe et les probabilités associées. Les erreurs de validation renvoient `422`, tandis qu'un modèle indisponible renvoie `503`. La route `/health` contrôle l'état du modèle et `/train` permet de réentraîner puis de sauvegarder le pipeline. Les requêtes, statuts et durées sont tracés dans `logs/api.log`.

- 🟢 **Vert** : réponses nominales `200` (`/health`, `/predict` et `/train`).
- 🔴 **Rouge** : erreurs fonctionnelles ou techniques (`422` et `503`).
- 🔵 **Bleu** : étapes de traitement de l'API et du pipeline de données.
- 🟡 **Jaune** : routes ou contrôles conditionnels qui orientent le flux.

---

*Schéma produit par Romain, 18/09/2026, dans le cadre de la certification Atlas.*
