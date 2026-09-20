"""Schémas Pydantic d'entrée et sortie de l'API.

L'API attend les 8 features décrivant le demandeur d'emploi (hors `usager_id` qui n'entre pas dans le modèle)
et retourne la classe prédite + les probabilités associées.
Seul `niveau_diplome` est autorisée à être manquante
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


NiveauDiplome = Literal["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
IndicateurBinaire = Literal["0", "1"]
RetourEmploi = Literal["bas", "moyen", "long"]


class Demandeur(BaseModel):
    """Données d'entrée d'une prédiction de retour à l'emploi.

    Les bornes des champs reflètent les plages observées dans le dataset
    d'entraînement et servent de garde-fou contre les entrées aberrantes.
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    age: int = Field(
        ge=18,
        le=63,
        description="Âge observé dans le dataset : 18 à 63 ans"
    )
    niveau_diplome: NiveauDiplome | None = Field(
        None,
        description="Valeurs possibles : Sans diplôme, Bac, Bac+2 ou Bac+5",
    )
    anciennete_poste_ans: float = Field(
        ge=0,
        le=30,
        description="Ancienneté observée dans le dataset : 0 à 30 ans",
    )
    code_rome_vise: str = Field(
        ...,
        min_length=5,
        max_length=5,
        pattern=r"^[A-Z][0-9]{4}$",
        description="Code ROME à 5 caractères",
    )
    code_insee_commune: str = Field(
        ...,
        min_length=4,
        max_length=5,
        pattern=r"^(?:[0-9]{4,5}|2[AB][0-9]{3})$",
        description="Code INSEE observé sur 4 ou 5 caractères",
    )
    est_allocataire: IndicateurBinaire = Field(
        ..., description="Valeurs possibles : 0 ou 1",
    )
    nationalite_hors_ue: IndicateurBinaire = Field(
        ..., description="Valeurs possibles : 0 ou 1",
    )
    synthese_entretien: str = Field(
        ..., min_length=1, description="Synthèse non vide de l'entretien",
    )

    @model_validator(mode="after")
    def validate_age_and_experience(self) -> "Demandeur":
        if self.age - self.anciennete_poste_ans <= 15:
            raise ValueError(
                "L'ancienneté est incompatible avec l'âge : le début d'activité doit être postérieur à 15 ans."
            )
        return self


class PredictionResponse(BaseModel):
    """Sortie d'une prédiction : classe + probabilités."""

    retour_emploi: RetourEmploi = Field(description="Classe prédite par le modèle.")
    probabilites: dict[RetourEmploi, float] = Field(
        description="Probabilité par classe (somme = 1.0).",
    )


class HealthResponse(BaseModel):
    """Sortie de la route /health."""

    status: Literal["ok", "degraded"] = Field(description="Statut global du service.")
    model_loaded: bool = Field(description="Vrai si le modèle est chargé en mémoire.")


class FeedbackCorrection(Demandeur):
    """Correction d'un conseiller sur une prédiction du modèle, réinjectée au réentraînement."""

    classe_predite: RetourEmploi = Field(description="Classe initialement prédite par le modèle.")
    classe_corrigee: RetourEmploi = Field(description="Classe corrigée par le conseiller.")
    commentaire: str | None = Field(
        None, max_length=500, description="Commentaire libre du conseiller.",
    )


class FeedbackResponse(BaseModel):
    """Accusé de réception d'une correction conseiller."""

    status: Literal["recorded"] = Field(description="Statut de l'enregistrement.")
    total_feedback_rows: int = Field(description="Nombre total de corrections enregistrées à ce jour.")


class TrainResponse(BaseModel):
    """Résultat d'un réentraînement monitoré du modèle."""

    status: Literal["trained", "rejected"] = Field(
        description="'trained' si le nouveau modèle a été promu, 'rejected' s'il a été refusé après validation.",
    )
    model_path: str = Field(description="Chemin du modèle sauvegardé.")
    training_rows: int = Field(description="Nombre de lignes utilisées pour l'entraînement (dataset + feedback).")
    feedback_rows_used: int = Field(description="Nombre de corrections conseillers intégrées.")
    metrics: dict[str, float] = Field(description="Métriques de validation (accuracy, f1_macro).")
    promoted: bool = Field(description="Vrai si le nouveau modèle a remplacé le modèle en production.")