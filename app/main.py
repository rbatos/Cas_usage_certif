"""API FastAPI pour la prédiction et l'entraînement du modèle emploi."""

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import json
import joblib
import sys
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.openapi.docs import get_redoc_html
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from app.middleware import RequestLoggingMiddleware
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

from loguru import logger


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "modele_lgbm_v1.0_S1_multimodale_complete.joblib"
DATA_PATH = ROOT_DIR / "data" / (
    "dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas "
    "- mai-oct2026 (Session-00279143).csv"
)
FEEDBACK_PATH = ROOT_DIR / "data" / "feedback_conseillers.csv"
HISTORY_PATH = ROOT_DIR / "data" / "historique_inferences.csv"
BASELINE_METRICS_PATH = ROOT_DIR / "models" / "train_metrics_baseline.json"
FEEDBACK_COLUMNS = [
    "age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise",
    "code_insee_commune", "est_allocataire", "nationalite_hors_ue",
    "synthese_entretien", "classe_predite", "classe_corrigee", "commentaire",
]
HISTORY_COLUMNS = [
    "horodatage", "request_id", "conseiller_id", "age", "niveau_diplome",
    "code_rome_vise", "departement", "retour_emploi", "probabilite_max",
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
    rotation="10 MB",       # nouveau fichier à 10 Mo
    retention="7 days",     # garde 7 jours d'historique
    compression="gz",       # compresse les anciens fichiers
    format=log_format,
    enqueue=True,           # thread-safe
    level="INFO",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, le libère à l'arrêt.

    Politique fail-fast : si `load_model` échoue, l'exception remonte,
    Uvicorn s'arrête et le conteneur sort en erreur (pas d'API zombie
    qui répondrait OK avec un modèle cassé ou absent).
    """
    global pipeline_lgbm
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
    """Charge le pipeline complet, prétraitement inclus.

    Politique fail-fast : toute erreur (fichier absent, import impossible,
    mémoire insuffisante...) est journalisée puis relevée telle quelle,
    pour que l'appelant (le lifespan au démarrage) laisse l'exception
    remonter et stoppe l'API plutôt que de continuer avec un modèle cassé.

    Raises:
        FileNotFoundError: Si le fichier modèle est absent.
        OSError | EOFError | ImportError | AttributeError | ValueError | MemoryError:
            Si le désérialisation du modèle échoue.
    """
    global model_load_error, pipeline_lgbm
    logger.info("Chargement du modèle: {}", MODEL_PATH)
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
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


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
        probabilites={
            CLASS_LABELS[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    )
    logger.info(
        "Prédiction retournée: retour_emploi={} probabilites={}",
        response.retour_emploi,
        response.probabilites,
    )
    save_history_entry(request, response, x_conseiller_id, http_request.state.request_id)
    return response


def save_history_entry(
    request: Demandeur, response: PredictionResponse, conseiller_id: str | None, request_id: str,
) -> None:
    """Ajoute une ligne à l'historique persistant des inférences.

    Échoue silencieusement (journalisé) plutôt que de faire échouer la prédiction :
    l'historique est une fonctionnalité de confort, pas critique pour la réponse.
    """
    row = pd.DataFrame([{
        "horodatage": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": request_id,
        "conseiller_id": conseiller_id or "inconnu",
        "age": request.age,
        "niveau_diplome": request.niveau_diplome,
        "code_rome_vise": request.code_rome_vise,
        "departement": str(request.code_insee_commune)[:2],
        "retour_emploi": response.retour_emploi,
        "probabilite_max": round(max(response.probabilites.values()), 3),
    }])
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

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[diploma_order])),
    ])
    text_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("to_1d", FunctionTransformer(flatten_text, validate=False)),
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
        ("ord", ordinal_pipeline, ordinal_columns),
        ("txt", text_pipeline, text_columns),
    ])
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
        "usager_id", "age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise",
        "code_insee_commune", "est_allocataire", "nationalite_hors_ue",
        "synthese_entretien", "classe_retour_emploi",
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
    global pipeline_lgbm
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
            features, data[target], test_size=0.2, random_state=42, stratify=data[target],
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

        if promoted:
            final_pipeline = build_training_pipeline()
            final_pipeline.fit(features, data[target])
            with MODEL_LOCK:
                joblib.dump(final_pipeline, MODEL_PATH)
                pipeline_lgbm = final_pipeline
            BASELINE_METRICS_PATH.write_text(json.dumps(metrics))
            logger.info("Nouveau modèle promu: {} lignes, modèle sauvegardé", len(features))
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
    )