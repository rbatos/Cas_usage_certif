"""Suivi MLflow des entraînements et promotion contrôlée des modèles."""

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parents[1]
# Configuration MLflow par défaut et récupération des variables d'environnement.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", (ROOT_DIR / "mlruns").as_uri())
# Nom de l'expérience MLflow, nom du modèle et alias par défaut.
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "orientation-retour-emploi")
# Alias par défaut pour le modèle dans le registre MLflow.
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "orientation-retour-emploi")
# Alias par défaut pour le modèle promu dans le registre MLflow.
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def configure_mlflow() -> None:
    """Configure le stockage local ou distant défini par l'environnement."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_training_run(
    pipeline,
    metrics: dict[str, float],
    baseline_f1: float | None,
    promoted: bool,
    training_rows: int,
    feedback_rows: int,
):
    """Trace un entraînement et enregistre un modèle accepté dans le registre MLflow.

    Args:
        pipeline: Pipeline scikit-learn entraîné.
        metrics: Dictionnaire des métriques de validation.
        baseline_f1: F1 macro du modèle de référence, si disponible.
        promoted: Vrai si le modèle a été promu.
        training_rows: Nombre de lignes utilisées pour l'entraînement.
        feedback_rows: Nombre de corrections conseillers intégrées.

    Returns:
        Tuple contenant l'ID du run MLflow et la version du modèle enregistré (ou None si non promu).
    """
    configure_mlflow()
    model_parameters = pipeline.named_steps["modele"].get_params()
    tracked_parameters = {
        key: value
        for key, value in model_parameters.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    tracked_parameters.update(
        {
            "training_rows": training_rows,
            "feedback_rows": feedback_rows,
        }
    )

    with mlflow.start_run() as run:
        mlflow.log_params(tracked_parameters)
        mlflow.log_metrics(metrics)
        if baseline_f1 is not None:
            mlflow.log_metric("baseline_f1_macro", baseline_f1)
        mlflow.set_tags(
            {
                "promotion_status": "promoted" if promoted else "rejected",
                "model_alias": MLFLOW_MODEL_ALIAS,
            }
        )

        model_version = None
        if promoted:
            mlflow.sklearn.log_model(pipeline, "model")
            registered = mlflow.register_model(
                f"runs:/{run.info.run_id}/model",
                MLFLOW_MODEL_NAME,
            )
            client = MlflowClient()
            client.set_registered_model_alias(
                MLFLOW_MODEL_NAME,
                MLFLOW_MODEL_ALIAS,
                registered.version,
            )
            model_version = str(registered.version)
            mlflow.set_tag("registered_model_version", model_version)

        return run.info.run_id, model_version
