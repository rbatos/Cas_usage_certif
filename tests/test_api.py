"""Tests des routes FastAPI."""

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
