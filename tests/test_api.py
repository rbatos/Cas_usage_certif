"""Tests des routes FastAPI."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main


class FakePipeline:
    """Pipeline minimal pour tester la route sans charger le modèle réel."""

    def predict_proba(self, features):
        assert list(features.columns) == [
            "age",
            "niveau_diplome",
            "anciennete_poste_ans",
            "code_rome_vise",
            "est_allocataire",
            "nationalite_hors_ue",
            "synthese_entretien",
            "departement",
        ]
        return [[0.1, 0.7, 0.2]]

    def predict(self, features):
        return [1]


@pytest.fixture
def client(monkeypatch):
    """Client API avec un pipeline de prédiction contrôlé.

    Le lifespan appelle load_model() au démarrage : on le neutralise pour
    que le FakePipeline ne soit pas écrasé par un chargement réel du modèle.
    """
    monkeypatch.setattr(main, "load_model", lambda: None)
    monkeypatch.setattr(main, "pipeline_lgbm", FakePipeline())
    monkeypatch.setattr(main, "model_load_error", None)
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Payload accepté par le schéma Pydantic."""
    return {
        "age": 35,
        "niveau_diplome": "Bac+2",
        "anciennete_poste_ans": 10,
        "code_rome_vise": "A1101",
        "code_insee_commune": "75056",
        "est_allocataire": "1",
        "nationalite_hors_ue": "0",
        "synthese_entretien": "Expérience en gestion de projet.",
    }


def test_health_returns_loaded_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
    assert response.headers["X-Request-ID"]


def test_predict_returns_prediction_and_probabilities(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "retour_emploi": "moyen",
        "probabilites": {"bas": 0.1, "moyen": 0.7, "long": 0.2},
    }


def test_predict_rejects_inconsistent_age_and_experience(client, valid_payload):
    valid_payload["anciennete_poste_ans"] = 25

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert "incompatible avec l'âge" in response.json()["detail"][0]["msg"]


def test_predict_rejects_unknown_fields(client, valid_payload):
    valid_payload["champ_inconnu"] = "interdit"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_request_id_is_preserved(client, valid_payload):
    request_id = "test-request-123"

    response = client.post(
        "/predict",
        json=valid_payload,
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id


@pytest.fixture
def feedback_payload():
    """Payload accepté par le schéma FeedbackCorrection."""
    return {
        "age": 35,
        "niveau_diplome": "Bac+2",
        "anciennete_poste_ans": 10,
        "code_rome_vise": "A1101",
        "code_insee_commune": "75056",
        "est_allocataire": "1",
        "nationalite_hors_ue": "0",
        "synthese_entretien": "Expérience en gestion de projet.",
        "classe_predite": "moyen",
        "classe_corrigee": "long",
        "commentaire": "Reclassé après entretien approfondi",
    }


@pytest.fixture
def isolated_feedback_path(tmp_path, monkeypatch):
    """Redirige FEEDBACK_PATH vers un fichier temporaire pour ne pas polluer les données réelles."""
    path = tmp_path / "feedback_conseillers.csv"
    monkeypatch.setattr(main, "FEEDBACK_PATH", path)
    return path


def test_feedback_records_correction_and_counts_rows(client, feedback_payload, isolated_feedback_path):
    """Vérifie que le feedback est correctement enregistré et que le nombre de lignes est compté."""
    response = client.post("/feedback", json=feedback_payload)

    assert response.status_code == 200
    assert response.json() == {"status": "recorded", "total_feedback_rows": 1}
    assert isolated_feedback_path.exists()


def test_feedback_appends_multiple_rows(client, feedback_payload, isolated_feedback_path):
    """Vérifie que plusieurs feedbacks sont correctement ajoutés et comptés."""
    client.post("/feedback", json=feedback_payload)
    response = client.post("/feedback", json=feedback_payload)

    assert response.status_code == 200
    assert response.json()["total_feedback_rows"] == 2


def test_feedback_rejects_unknown_fields(client, feedback_payload, isolated_feedback_path):
    """Vérifie que le feedback avec des champs inconnus est rejeté."""
    feedback_payload["champ_inconnu"] = "interdit"

    response = client.post("/feedback", json=feedback_payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.fixture
def isolated_history_path(tmp_path, monkeypatch):
    """Redirige HISTORY_PATH vers un fichier temporaire pour ne pas polluer les données réelles."""
    path = tmp_path / "historique_inferences.csv"
    monkeypatch.setattr(main, "HISTORY_PATH", path)
    return path


def test_history_returns_empty_list_when_no_file(client, isolated_history_path):
    """Vérifie que l'historique renvoie une liste vide si aucune prédiction n'a encore été faite."""
    response = client.get("/history")

    assert response.status_code == 200
    assert response.json() == {"entries": []}


def test_history_records_entry_after_predict(client, valid_payload, isolated_history_path):
    """Vérifie qu'une prédiction est bien journalisée et retrouvable via /history."""
    client.post("/predict", json=valid_payload)

    response = client.get("/history")

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["retour_emploi"] == "moyen"
    assert entries[0]["conseiller_id"] == "inconnu"
    assert entries[0]["departement"] == "75"


def test_history_filters_by_conseiller_id(client, valid_payload, isolated_history_path):
    """Vérifie que le filtre conseiller_id ne renvoie que les prédictions de ce conseiller."""
    client.post("/predict", json=valid_payload, headers={"X-Conseiller-ID": "romain"})
    client.post("/predict", json=valid_payload, headers={"X-Conseiller-ID": "julien"})

    response = client.get("/history", params={"conseiller_id": "romain"})

    entries = response.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["conseiller_id"] == "romain"


def test_history_respects_limit(client, valid_payload, isolated_history_path):
    """Vérifie que le paramètre limit borne le nombre de lignes retournées."""
    for _ in range(3):
        client.post("/predict", json=valid_payload)

    response = client.get("/history", params={"limit": 2})

    assert len(response.json()["entries"]) == 2


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    """Mini dataset d'entraînement synthétique, équilibré sur les 3 classes."""
    diplomes = ["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
    rows = [{
        "usager_id": f"ID_{index:04d}",
        "age": 20 + (index % 30),
        "niveau_diplome": diplomes[index % 4],
        "anciennete_poste_ans": 1 + (index % 10),
        "code_rome_vise": "A1101",
        "code_insee_commune": "75056",
        "est_allocataire": index % 2,
        "nationalite_hors_ue": index % 2,
        "synthese_entretien": "Recherche active de travail avec mobilité géographique",
        "classe_retour_emploi": index % 3,
    } for index in range(30)]
    return pd.DataFrame(rows)


@pytest.fixture
def isolated_train_paths(tmp_path, monkeypatch, synthetic_dataset):
    """Redirige DATA_PATH, MODEL_PATH, BASELINE_METRICS_PATH et FEEDBACK_PATH vers tmp_path."""
    data_path = tmp_path / "dataset.csv"
    synthetic_dataset.to_csv(data_path, index=False)
    monkeypatch.setattr(main, "DATA_PATH", data_path)
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(main, "BASELINE_METRICS_PATH", tmp_path / "baseline.json")
    monkeypatch.setattr(main, "FEEDBACK_PATH", tmp_path / "feedback_conseillers.csv")


def test_train_returns_404_when_dataset_missing(client, tmp_path, monkeypatch):
    """Vérifie que /train échoue proprement si le dataset local est introuvable."""
    monkeypatch.setattr(main, "DATA_PATH", tmp_path / "absent.csv")

    response = client.post("/train")

    assert response.status_code == 404


def test_train_promotes_model_when_no_baseline(client, isolated_train_paths):
    """Sans baseline existante, le nouveau modèle est promu et sauvegardé."""
    response = client.post("/train")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "trained"
    assert data["promoted"] is True
    assert main.MODEL_PATH.exists()
    assert main.BASELINE_METRICS_PATH.exists()


def test_train_rejects_when_below_baseline(client, isolated_train_paths):
    """Un nouveau modèle sous le seuil de tolérance par rapport à la baseline est rejeté."""
    main.BASELINE_METRICS_PATH.write_text(json.dumps({"f1_macro": 0.99}))

    response = client.post("/train")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["promoted"] is False
    assert not main.MODEL_PATH.exists()
