"""API FastAPI pour la prédiction et l'entraînement du modèle emploi."""

from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from app.schemas import HealthResponse, PredictionResponse, TrainResponse, Demandeur


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "modele_lgbm_S1_multimodale_complete.joblib"
DATA_PATH = ROOT_DIR / "data" / (
    "dataset_trajectoire_emploi_Sujet Examen CISIA - Promo Upskilling Atlas "
    "- mai-oct2026 (Session-00279143).csv"
)

CLASS_LABELS = {0: "bas", 1: "moyen", 2: "long"}
MODEL_LOCK = Lock()
pipeline_lgbm = None
model_load_error = None

app = FastAPI(
    title="API orientation retour à l'emploi",
    version="1.0.0",
)


def flatten_text(values):
    """Transforme la colonne texte 2D en vecteur 1D pour TF-IDF."""
    return values.ravel()


def load_model() -> None:
    """Charge le pipeline complet, prétraitement inclus."""
    global model_load_error, pipeline_lgbm
    if not MODEL_PATH.exists():
        pipeline_lgbm = None
        model_load_error = f"Fichier modèle absent : {MODEL_PATH}"
        return
    try:
        pipeline_lgbm = joblib.load(MODEL_PATH)
        model_load_error = None
    except (OSError, EOFError, ImportError, AttributeError, ValueError) as error:
        pipeline_lgbm = None
        model_load_error = f"Chargement du modèle impossible : {error}"


def ensure_model_loaded() -> bool:
    """Recharge le modèle si le serveur n'a pas exécuté son démarrage."""
    if pipeline_lgbm is None:
        with MODEL_LOCK:
            if pipeline_lgbm is None:
                load_model()
    return pipeline_lgbm is not None


def request_to_features(request: Demandeur) -> pd.DataFrame:
    """Convertit le payload API dans le schéma attendu par le pipeline."""
    payload = request.model_dump()
    code_insee = str(payload.pop("code_insee_commune"))
    payload["departement"] = code_insee[:2]
    for column in ("est_allocataire", "nationalite_hors_ue"):
        if isinstance(payload[column], str) and payload[column].isdigit():
            payload[column] = int(payload[column])
    return pd.DataFrame([payload])

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = ensure_model_loaded()
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Demandeur) -> PredictionResponse:
    if not ensure_model_loaded():
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Modèle indisponible",
        )

    features = request_to_features(request)
    with MODEL_LOCK:
        probabilities = pipeline_lgbm.predict_proba(features)[0]
        predicted_class = int(pipeline_lgbm.predict(features)[0])

    return PredictionResponse(
        retour_emploi=CLASS_LABELS[predicted_class],
        probabilites={
            CLASS_LABELS[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    )


def build_training_pipeline() -> Pipeline:
    """Reconstruit le même prétraitement que celui utilisé pour le modèle S1."""
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


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    global pipeline_lgbm
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Jeu de données introuvable")

    try:
        data = pd.read_csv(DATA_PATH)
        target = "classe_retour_emploi"
        if target not in data.columns:
            raise ValueError(f"Colonne cible absente : {target}")

        features = data.drop(columns=[target]).copy()
        features["departement"] = features["code_insee_commune"].astype(str).str[:2]
        features = features.drop(columns=["code_insee_commune", "usager_id"], errors="ignore")
        if "age" in features and "anciennete_poste_ans" in features:
            inconsistent = (features["age"] - features["anciennete_poste_ans"]) <= 15
            features.loc[inconsistent, "anciennete_poste_ans"] = pd.NA

        new_pipeline = build_training_pipeline()
        with MODEL_LOCK:
            new_pipeline.fit(features, data[target])
            joblib.dump(new_pipeline, MODEL_PATH)
            pipeline_lgbm = new_pipeline
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=f"Entraînement impossible : {error}") from error

    return TrainResponse(
        status="trained",
        model_path=str(MODEL_PATH),
        training_rows=len(features),
    )