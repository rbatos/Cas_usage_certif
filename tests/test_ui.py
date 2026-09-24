"""Tests de l'interface Streamlit."""

from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "services" / "ui-streamlit" / "app.py"


class FakeResponse:
    """Réponse HTTP minimale utilisée par l'interface pendant le test."""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeHealthClient:
    """Client minimal simulant la route de santé affichée dans la barre latérale."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, path):
        return FakeResponse({"model_loaded": True, "model_version": "1.0"})


def test_feedback_form_sends_last_prediction_and_displays_confirmation(monkeypatch):
    sent_payload = {}

    def fake_post(url, json, timeout):
        assert url.endswith("/feedback")
        sent_payload.update(json)
        return FakeResponse({"status": "recorded", "total_feedback_rows": 3})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: FakeHealthClient())

    prediction_payload = {
        "age": 35,
        "niveau_diplome": "Bac+2",
        "anciennete_poste_ans": 10.0,
        "code_rome_vise": "M1805",
        "code_insee_commune": "75056",
        "est_allocataire": "1",
        "nationalite_hors_ue": "0",
        "synthese_entretien": "Recherche active.",
    }
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["derniere_prediction"] = {
        "payload": prediction_payload,
        "classe_predite": "moyen",
    }
    app.session_state["feedback_enregistre"] = False
    app.run(timeout=10)

    next(widget for widget in app.selectbox if widget.label == "Classe réellement constatée").select("long")
    next(widget for widget in app.text_area if widget.label == "Commentaire facultatif").input(
        "Situation réévaluée après suivi."
    )
    next(button for button in app.button if button.label == "Enregistrer la correction").click()
    app.run(timeout=10)

    assert sent_payload == {
        **prediction_payload,
        "classe_predite": "moyen",
        "classe_corrigee": "long",
        "commentaire": "Situation réévaluée après suivi.",
    }
    assert app.session_state["feedback_enregistre"] is True
    assert any("Total : 3 correction(s)." in message.value for message in app.success)


def test_training_button_calls_train_and_displays_result(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(
            {
                "status": "trained",
                "model_path": "models/model.joblib",
                "training_rows": 42,
                "feedback_rows_used": 3,
                "metrics": {"accuracy": 0.8, "f1_macro": 0.75},
                "promoted": True,
                "artifact_version": "1.1",
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: FakeHealthClient())

    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=10)

    next(button for button in app.button if button.label == "Lancer le réentraînement").click()
    app.run(timeout=10)

    assert calls == [("http://localhost:8000/train", {"timeout": 120})]
    assert any("modèle promu en production" in message.value for message in app.success)
    assert any("42 ligne(s) utilisées" in message.value for message in app.success)