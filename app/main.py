"""API FastAPI pour la prédiction et l'entraînement du modèle emploi."""

import hashlib
import json
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.openapi.docs import get_redoc_html
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from app.middleware import RequestLoggingMiddleware
from app.mlflow_tracking import log_training_run
from app.schemas import (
    Demandeur,
    FeedbackCorrection,
    FeedbackResponse,
    HealthResponse,
    HistoryEntry,
    HistoryResponse,
    PredictionResponse,
    TrainResponse,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "modele_lgbm_v1.0_S1_multimodale_complete.joblib"
MODEL_MANIFEST_PATH = ROOT_DIR / "models" / "model_manifest.json"
MODEL_VERSION = "1.0"
MODEL_FILENAME_TEMPLATE = "modele_lgbm_v{version}_S1_multimodale_complete.joblib"
MODEL_METADATA_TEMPLATE = "modele_lgbm_v{version}_S1_multimodale_complete_metadata.json"
DATA_PATH = (
    ROOT_DIR
    / "data"
    / ("dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas " "- mai-oct2026 (Session-00279143).csv")
)
FEEDBACK_PATH = ROOT_DIR / "data" / "feedback_conseillers.csv"
HISTORY_PATH = ROOT_DIR / "data" / "historique_inferences.csv"
BASELINE_METRICS_PATH = ROOT_DIR / "models" / "train_metrics_baseline.json"
FEEDBACK_COLUMNS = [
    "age",
    "niveau_diplome",
    "anciennete_poste_ans",
    "code_rome_vise",
    "code_insee_commune",
    "est_allocataire",
    "nationalite_hors_ue",
    "synthese_entretien",
    "classe_predite",
    "classe_corrigee",
    "commentaire",
]
HISTORY_COLUMNS = [
    "horodatage",
    "request_id",
    "conseiller_id",
    "age",
    "niveau_diplome",
    "code_rome_vise",
    "departement",
    "retour_emploi",
    "probabilite_max",
]
# Tolérance de dégradation du f1_macro avant de refuser la promotion d'un nouveau modèle
DEGRADATION_TOLERANCE = 0.02

CLASS_LABELS = {0: "bas", 1: "moyen", 2: "long"}
REVERSE_CLASS_LABELS = {label: index for index, label in CLASS_LABELS.items()}
MODEL_LOCK = Lock()
FEEDBACK_LOCK = Lock()
HISTORY_LOCK = Lock()
pipeline_lgbm = None
model_load_error = None


def read_model_manifest() -> dict | None:
    """Retourne le manifeste du modèle actif, s'il existe et est valide."""
    if not MODEL_MANIFEST_PATH.exists():
        return None
    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        model_path = Path(manifest["model_path"])
        if model_path.is_absolute() or ".." in model_path.parts:
            raise ValueError("Le chemin du modèle actif doit rester relatif au dossier models")
        if not manifest.get("model_version"):
            raise ValueError("La version du modèle actif est absente")
        return manifest
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Manifeste du modèle invalide : {error}") from error


def resolve_active_model() -> tuple[Path, str]:
    """Résout le chemin et la version du modèle à charger."""
    manifest = read_model_manifest()
    if manifest is None:
        return MODEL_PATH, MODEL_VERSION
    return ROOT_DIR / "models" / manifest["model_path"], str(manifest["model_version"])


def next_model_version() -> str:
    """Calcule la prochaine version patch à partir du modèle actif et des artefacts présents."""
    model_directory = MODEL_MANIFEST_PATH.parent
    versions = []
    manifest = read_model_manifest()
    if manifest is not None:
        versions.append(str(manifest["model_version"]))
    versions.extend(
        match.group(1)
        for path in model_directory.glob("modele_lgbm_v*_S1_multimodale_complete.joblib")
        if (match := re.match(r"modele_lgbm_v(\d+\.\d+)_S1_multimodale_complete\.joblib", path.name))
    )
    major, minor = max((tuple(map(int, version.split("."))) for version in versions), default=(1, 0))
    return f"{major}.{minor + 1}"


def write_promoted_model_metadata(
    model_path: Path,
    version: str,
    metrics: dict[str, float],
    training_rows: int,
    pipeline: Pipeline,
    feature_columns: list[str],
) -> Path:
    """Écrit les metadata de l'artefact effectivement promu."""
    metadata_path = model_path.with_name(MODEL_METADATA_TEMPLATE.format(version=version))
    try:
        dataset_source = str(DATA_PATH.parent.relative_to(ROOT_DIR))
    except ValueError:
        dataset_source = str(DATA_PATH.parent)
    metadata = {
        "modele": "modele_lgbm",
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": "S1_multimodale_complete",
        "chemin": str(model_path.relative_to(MODEL_MANIFEST_PATH.parent)),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "dataset": DATA_PATH.name,
        "dataset_source": dataset_source,
        "hyperparameters": pipeline.named_steps["modele"].get_params(),
        "metrics": metrics,
        "training_rows": training_rows,
        "features_columns": feature_columns,
        "target_column": "classe_retour_emploi",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def write_model_manifest(model_path: Path, version: str, metadata_path: Path, mlflow_version: str | None) -> None:
    """Publie atomiquement la nouvelle référence du modèle actif."""
    manifest = {
        "model_version": version,
        "model_path": str(model_path.relative_to(MODEL_MANIFEST_PATH.parent)),
        "metadata_path": str(metadata_path.relative_to(MODEL_MANIFEST_PATH.parent)),
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "mlflow_model_version": mlflow_version,
    }
    temporary_path = MODEL_MANIFEST_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(MODEL_MANIFEST_PATH)

# --- Loguru configuration ---------------------------------------------------
# Configuration Loguru (au démarrage du module)
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()  # vire le handler par défaut


def log_format(record):
    """Format compact qui supporte les messages contenant des dictionnaires."""
    request_id = record["extra"].get("request_id", "-")
    timestamp = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    level = record["level"].name.ljust(8)
    message = record["message"].replace("{", "{{").replace("}", "}}")
    return f"<green>{timestamp}</green> | <level>{level}</level> | request_id={request_id} | {message}\n"


logger.add(sys.stderr, level="INFO", colorize=True, format=log_format)
# Configuration du log rotate
logger.add(
    LOGS_DIR / "api.log",
    rotation="10 MB",  # nouveau fichier à 10 Mo
    retention="7 days",  # garde 7 jours d'historique
    compression="gz",  # compresse les anciens fichiers
    format=log_format,
    enqueue=True,  # thread-safe
    level="INFO",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, le libère à l'arrêt.

    Politique fail-fast : si `load_model` échoue, l'exception remonte,
    Uvicorn s'arrête et le conteneur sort en erreur (pas d'API zombie
    qui répondrait OK avec un modèle cassé ou absent).
    """
    global pipeline_lgbm, MODEL_PATH, MODEL_VERSION
    load_model()
    yield
    pipeline_lgbm = None
    logger.info("Modèle libéré à l'arrêt de l'API")


app = FastAPI(
    title="API orientation retour à l'emploi",
    version="1.0.0",
    redoc_url=None,
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/redoc", include_in_schema=False)
def redoc_documentation():
    """Expose ReDoc avec une version stable du bundle JavaScript."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js",
    )


def flatten_text(values):
    """Transforme la colonne texte 2D en vecteur 1D pour TF-IDF.

    Args:
        values (np.ndarray): Colonne texte 2D.

    Returns:
        np.ndarray: Vecteur 1D aplati.
    """
    return values.ravel()


def load_model() -> None:
    """Charge le modèle actif en mémoire.

    Cette fonction met à jour les variables globales `pipeline_lgbm`,
    `MODEL_PATH` et `MODEL_VERSION`. En cas d'erreur, elle journalise
    l'erreur et relance l'exception.

    Raises:
        FileNotFoundError: Si le fichier modèle est absent.
        ValueError: Si le hash du modèle actif ne correspond pas au manifeste.
        OSError | EOFError | ImportError | AttributeError | ValueError | MemoryError:
            Si le désérialisation du modèle échoue.
    """
    global model_load_error, pipeline_lgbm, MODEL_PATH, MODEL_VERSION
    try:
        manifest = read_model_manifest()
        active_model_path, active_model_version = resolve_active_model()
        if manifest is not None and manifest.get("sha256"):
            actual_hash = hashlib.sha256(active_model_path.read_bytes()).hexdigest()
            if actual_hash != manifest["sha256"]:
                raise ValueError(f"Hash du modèle actif invalide : {active_model_path}")
    except ValueError as error:
        model_load_error = str(error)
        logger.error(model_load_error)
        raise
    MODEL_PATH = active_model_path
    MODEL_VERSION = active_model_version
    logger.info("Chargement du modèle v{}: {}", MODEL_VERSION, MODEL_PATH)
    if not MODEL_PATH.exists():
        model_load_error = f"Fichier modèle absent : {MODEL_PATH}"
        logger.error("Modèle introuvable: {}", MODEL_PATH)
        raise FileNotFoundError(model_load_error)
    try:
        pipeline_lgbm = joblib.load(MODEL_PATH)
        model_load_error = None
        logger.info("Modèle chargé avec succès")
    except (OSError, EOFError, ImportError, AttributeError, ValueError, MemoryError) as error:
        model_load_error = f"Chargement du modèle impossible : {error}"
        logger.exception("Échec du chargement du modèle")
        raise


def ensure_model_loaded() -> bool:
    """Recharge le modèle si le serveur n'a pas exécuté son démarrage.

    Contrairement au lifespan, on ne fait pas remonter l'exception ici :
    une requête isolée doit se solder par un 503, pas par un crash du worker.

    Returns:
        bool: True si le modèle est chargé en mémoire, False sinon.
    """
    if pipeline_lgbm is None:
        logger.debug("Modèle absent de la mémoire, tentative de chargement")
        with MODEL_LOCK:
            if pipeline_lgbm is None:
                try:
                    load_model()
                except Exception:
                    return False
    return pipeline_lgbm is not None


def request_to_features(request: Demandeur) -> pd.DataFrame:
    """Convertit le payload API dans le schéma attendu par le pipeline.

    Args:
        request (Demandeur): Objet représentant la requête API.

    Returns:
        pd.DataFrame: DataFrame contenant les features prêtes pour la prédiction.
    """
    logger.debug("Préparation des features pour la prédiction")
    payload = request.model_dump()
    code_insee = str(payload.pop("code_insee_commune"))
    payload["departement"] = code_insee[:2]
    for column in ("est_allocataire", "nationalite_hors_ue"):
        if isinstance(payload[column], str) and payload[column].isdigit():
            payload[column] = int(payload[column])
    return pd.DataFrame([payload])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Point de terminaison pour vérifier l'état de santé de l'API.

    Returns:
        HealthResponse: Objet contenant le statut de santé et l'état de chargement du modèle.
    """
    loaded = ensure_model_loaded()
    logger.info("Contrôle de santé: model_loaded={}", loaded)
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackCorrection) -> FeedbackResponse:
    """Enregistre la correction d'un conseiller pour le réentraînement monitoré.

    Args:
        feedback (FeedbackCorrection): Caractéristiques du dossier, classe prédite et classe corrigée.

    Returns:
        FeedbackResponse: Accusé de réception avec le total de corrections stockées.
    """
    row = pd.DataFrame([{column: getattr(feedback, column) for column in FEEDBACK_COLUMNS}])
    with FEEDBACK_LOCK:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(FEEDBACK_PATH, mode="a", header=not FEEDBACK_PATH.exists(), index=False)
        total_rows = len(pd.read_csv(FEEDBACK_PATH))
    logger.info(
        "Feedback conseiller enregistré: classe_predite={} classe_corrigee={}",
        feedback.classe_predite,
        feedback.classe_corrigee,
    )
    return FeedbackResponse(status="recorded", total_feedback_rows=total_rows)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: Demandeur,
    http_request: Request,
    x_conseiller_id: str | None = Header(default=None, alias="X-Conseiller-ID"),
) -> PredictionResponse:
    """Point de terminaison pour effectuer une prédiction.

    Args:
        request (Demandeur): Objet représentant la requête API.
        http_request (Request): Requête HTTP brute, pour récupérer le request_id du middleware.
        x_conseiller_id (str | None): Identifiant optionnel du conseiller, pour l'historique.

    Returns:
        PredictionResponse: Objet contenant la classe prédite et les probabilités par classe.
    """
    request_data = request.model_dump(exclude={"synthese_entretien"})
    request_data["synthese_entretien_length"] = len(request.synthese_entretien)
    logger.info("Paramètres validés pour la prédiction: {}", request_data)
    if not ensure_model_loaded():
        logger.error("Prédiction impossible: modèle indisponible")
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Modèle indisponible",
        )

    features = request_to_features(request)
    with MODEL_LOCK:
        probabilities = pipeline_lgbm.predict_proba(features)[0]
        predicted_class = int(pipeline_lgbm.predict(features)[0])

    response = PredictionResponse(
        retour_emploi=CLASS_LABELS[predicted_class],
        probabilites={CLASS_LABELS[index]: float(probability) for index, probability in enumerate(probabilities)},
    )
    logger.info(
        "Prédiction retournée: retour_emploi={} probabilites={}",
        response.retour_emploi,
        response.probabilites,
    )
    save_history_entry(request, response, x_conseiller_id, http_request.state.request_id)
    return response


def save_history_entry(
    request: Demandeur,
    response: PredictionResponse,
    conseiller_id: str | None,
    request_id: str,
) -> None:
    """Ajoute une ligne à l'historique persistant des inférences.

    Échoue silencieusement (journalisé) plutôt que de faire échouer la prédiction :
    l'historique est une fonctionnalité de confort, pas critique pour la réponse.
    """
    row = pd.DataFrame(
        [
            {
                "horodatage": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request_id": request_id,
                "conseiller_id": conseiller_id or "inconnu",
                "age": request.age,
                "niveau_diplome": request.niveau_diplome,
                "code_rome_vise": request.code_rome_vise,
                "departement": str(request.code_insee_commune)[:2],
                "retour_emploi": response.retour_emploi,
                "probabilite_max": round(max(response.probabilites.values()), 3),
            }
        ]
    )
    try:
        with HISTORY_LOCK:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            row.to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)
    except OSError:
        logger.exception("Échec de l'écriture de l'historique des inférences")


@app.get("/history", response_model=HistoryResponse)
def get_history(conseiller_id: str | None = None, limit: int = 50) -> HistoryResponse:
    """Retourne l'historique persistant des inférences, du plus récent au plus ancien.

    Args:
        conseiller_id (str | None): Si fourni, ne renvoie que les prédictions de ce conseiller.
        limit (int): Nombre maximal de lignes retournées (défaut 50).

    Returns:
        HistoryResponse: Liste des prédictions passées correspondant aux filtres.
    """
    if not HISTORY_PATH.exists():
        return HistoryResponse(entries=[])
    with HISTORY_LOCK:
        history = pd.read_csv(
            HISTORY_PATH,
            dtype={
                "request_id": "string",
                "conseiller_id": "string",
                "niveau_diplome": "string",
                "code_rome_vise": "string",
                "departement": "string",
                "retour_emploi": "string",
            },
        )
    if conseiller_id:
        history = history[history["conseiller_id"] == conseiller_id]
    history = history.sort_values("horodatage", ascending=False).head(max(limit, 0))
    history = history.where(pd.notna(history), None)
    entries = [HistoryEntry(**row) for row in history.to_dict(orient="records")]
    return HistoryResponse(entries=entries)


def build_training_pipeline() -> Pipeline:
    """Reconstruit le même prétraitement que celui utilisé pour le modèle S1.

    Returns:
        Pipeline: Pipeline de prétraitement et de modèle prêt pour l'entraînement.
    """
    numeric_columns = ["age", "anciennete_poste_ans"]
    categorical_columns = [
        "nationalite_hors_ue",
        "est_allocataire",
        "code_rome_vise",
        "departement",
    ]
    ordinal_columns = ["niveau_diplome"]
    text_columns = ["synthese_entretien"]
    diploma_order = ["Sans diplôme", "Bac", "Bac+2", "Bac+5"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[diploma_order])),
        ]
    )
    text_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("to_1d", FunctionTransformer(flatten_text, validate=False)),
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
            ("ord", ordinal_pipeline, ordinal_columns),
            ("txt", text_pipeline, text_columns),
        ]
    )
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=24,
        min_child_samples=10,
        colsample_bytree=0.7,
        random_state=42,
        verbosity=-1,
    )
    return Pipeline([("preparation", preprocessor), ("modele", model)])


def load_feedback_dataframe() -> pd.DataFrame:
    """Charge les corrections des conseillers au format du dataset d'entraînement.

    Returns:
        pd.DataFrame: Lignes de feedback prêtes à être fusionnées, vide si aucun feedback.
    """
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame()
    feedback = pd.read_csv(FEEDBACK_PATH)
    if feedback.empty:
        return feedback
    feedback = feedback.rename(columns={"classe_corrigee": "classe_retour_emploi"})
    feedback["classe_retour_emploi"] = feedback["classe_retour_emploi"].map(REVERSE_CLASS_LABELS)
    feedback["usager_id"] = [f"FEEDBACK_{index}" for index in feedback.index]
    training_columns = [
        "usager_id",
        "age",
        "niveau_diplome",
        "anciennete_poste_ans",
        "code_rome_vise",
        "code_insee_commune",
        "est_allocataire",
        "nationalite_hors_ue",
        "synthese_entretien",
        "classe_retour_emploi",
    ]
    return feedback[training_columns]


def load_baseline_f1() -> float | None:
    """Lit le f1_macro du modèle actuellement en production, s'il a été mesuré.

    Returns:
        float | None: f1_macro de référence, ou None si aucune baseline n'existe encore.
    """
    if not BASELINE_METRICS_PATH.exists():
        return None
    try:
        return json.loads(BASELINE_METRICS_PATH.read_text())["f1_macro"]
    except (OSError, ValueError, KeyError):
        return None


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    """Point de terminaison pour réentraîner le modèle, feedback conseillers inclus.

    Le nouveau modèle n'est promu en production que si ses métriques de
    validation ne se dégradent pas significativement par rapport à la
    baseline en place (garde-fou anti-régression).

    Returns:
        TrainResponse: Statut, chemin du modèle, lignes utilisées, métriques et décision de promotion.
    """
    global pipeline_lgbm, MODEL_PATH, MODEL_VERSION
    logger.info("Début de l'entraînement monitoré")
    if not DATA_PATH.exists():
        logger.error("Jeu de données introuvable: {}", DATA_PATH)
        raise HTTPException(status_code=404, detail="Jeu de données introuvable")

    try:
        data = pd.read_csv(DATA_PATH)
        feedback_data = load_feedback_dataframe()
        if not feedback_data.empty:
            data = pd.concat([data, feedback_data], ignore_index=True)
        logger.info(
            "Jeu de données chargé: {} lignes ({} issues du feedback conseillers)",
            len(data),
            len(feedback_data),
        )
        target = "classe_retour_emploi"
        if target not in data.columns:
            raise ValueError(f"Colonne cible absente : {target}")

        features = data.drop(columns=[target]).copy()
        features["departement"] = features["code_insee_commune"].astype(str).str[:2]
        features = features.drop(columns=["code_insee_commune", "usager_id"], errors="ignore")
        if "age" in features and "anciennete_poste_ans" in features:
            inconsistent = (features["age"] - features["anciennete_poste_ans"]) <= 15
            features.loc[inconsistent, "anciennete_poste_ans"] = pd.NA

        X_train, X_val, y_train, y_val = train_test_split(
            features,
            data[target],
            test_size=0.2,
            random_state=42,
            stratify=data[target],
        )
        validation_pipeline = build_training_pipeline()
        validation_pipeline.fit(X_train, y_train)
        predictions = validation_pipeline.predict(X_val)
        metrics = {
            "accuracy": float(accuracy_score(y_val, predictions)),
            "f1_macro": float(f1_score(y_val, predictions, average="macro")),
        }
        baseline_f1 = load_baseline_f1()
        promoted = baseline_f1 is None or metrics["f1_macro"] >= baseline_f1 - DEGRADATION_TOLERANCE
        logger.info("Métriques de validation: {} (baseline f1_macro={})", metrics, baseline_f1)

        tracked_pipeline = validation_pipeline
        model_version = None

        if promoted:
            final_pipeline = build_training_pipeline()
            final_pipeline.fit(features, data[target])
            tracked_pipeline = final_pipeline

        mlflow_run_id, model_version = log_training_run(
            pipeline=tracked_pipeline,
            metrics=metrics,
            baseline_f1=baseline_f1,
            promoted=promoted,
            training_rows=len(features),
            feedback_rows=len(feedback_data),
        )

        if promoted:
            promoted_version = next_model_version()
            promoted_path = MODEL_MANIFEST_PATH.parent / MODEL_FILENAME_TEMPLATE.format(version=promoted_version)
            promoted_metadata_path = promoted_path.with_name(
                MODEL_METADATA_TEMPLATE.format(version=promoted_version)
            )
            with MODEL_LOCK:
                joblib.dump(final_pipeline, promoted_path)
                write_promoted_model_metadata(
                    promoted_path,
                    promoted_version,
                    metrics,
                    len(features),
                    final_pipeline,
                    features.columns.tolist(),
                )
                write_model_manifest(promoted_path, promoted_version, promoted_metadata_path, model_version)
                MODEL_PATH = promoted_path
                MODEL_VERSION = promoted_version
                pipeline_lgbm = final_pipeline
            BASELINE_METRICS_PATH.write_text(json.dumps(metrics))
            logger.info(
                "Nouveau modèle promu: version {}, {} lignes, modèle sauvegardé",
                promoted_version,
                len(features),
            )
        else:
            logger.warning(
                "Modèle rejeté : f1_macro {} sous le seuil (baseline {} - tolérance {})",
                metrics["f1_macro"],
                baseline_f1,
                DEGRADATION_TOLERANCE,
            )
    except (OSError, ValueError, KeyError) as error:
        logger.exception("Échec de l'entraînement")
        raise HTTPException(status_code=422, detail=f"Entraînement impossible : {error}") from error

    return TrainResponse(
        status="trained" if promoted else "rejected",
        model_path=str(MODEL_PATH),
        training_rows=len(features),
        feedback_rows_used=len(feedback_data),
        metrics=metrics,
        promoted=promoted,
        mlflow_run_id=mlflow_run_id,
        model_version=model_version,
        artifact_version=MODEL_VERSION if promoted else None,
    )
