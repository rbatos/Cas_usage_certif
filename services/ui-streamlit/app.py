"""UI Streamlit — saisie conseiller pour la prédiction de retour à l'emploi.

Permet à un conseiller de saisir les 8 features d'un demandeur d'emploi
et d'obtenir la classe prédite (bas / moyen / long) ainsi que les
probabilités associées, via l'API FastAPI du projet (`app/main.py`).

L'URL de l'API est dans la variable d'environnement `API_URL`
(par défaut `http://localhost:8000`, ou `http://api:8000` sous docker-compose).
"""

from __future__ import annotations

import os

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL: str = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT_S = 10
TRAIN_TIMEOUT_S = 120

NIVEAUX_DIPLOME = ["Sans diplôme", "Bac", "Bac+2", "Bac+5"]
COULEUR_CLASSE = {"bas": st.success, "moyen": st.warning, "long": st.error}
ORDRE_CLASSES = ["bas", "moyen", "long"]

st.set_page_config(
    page_title="Orientation retour à l'emploi",
    page_icon="🧭",
    layout="centered",
)


def formater_erreur(detail) -> str:
    """Transforme un detail d'erreur FastAPI (str ou liste Pydantic) en texte lisible."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        lignes = []
        for erreur in detail:
            if isinstance(erreur, dict):
                champ = ".".join(str(p) for p in erreur.get("loc", []) if p != "body")
                message = erreur.get("msg", str(erreur))
                lignes.append(f"- **{champ}** : {message}" if champ else f"- {message}")
            else:
                lignes.append(f"- {erreur}")
        return "\n".join(lignes)
    return str(detail)


st.title("🧭 Orientation retour à l'emploi — saisie conseiller")
st.caption(
    "Renseigne les informations du demandeur d'emploi pour estimer son délai "
    "de retour à l'emploi (bas / moyen / long) et consulter les probabilités."
)

if "derniere_prediction" not in st.session_state:
    st.session_state.derniere_prediction = None
if "feedback_enregistre" not in st.session_state:
    st.session_state.feedback_enregistre = False

with st.sidebar:
    st.markdown("**Identifiant conseiller**")
    conseiller_id = st.text_input(
        "Identifiant conseiller",
        value="",
        placeholder="Ex : jdupont",
        label_visibility="collapsed",
        help="Utilisé pour retrouver ton historique de prédictions.",
    )
    lancer_entrainement = st.button(
        "Lancer le réentraînement",
        help="Réentraîne le modèle avec les corrections conseillers disponibles.",
    )

if lancer_entrainement:
    try:
        with st.spinner("Réentraînement en cours…"):
            reponse_entrainement = httpx.post(
                f"{API_URL}/train",
                timeout=TRAIN_TIMEOUT_S,
            )
            reponse_entrainement.raise_for_status()
            resultat_entrainement = reponse_entrainement.json()
    except httpx.ConnectError:
        st.error("Erreur de connexion : impossible de joindre l'API.")
    except httpx.TimeoutException:
        st.error(f"Réentraînement trop long : pas de réponse en {TRAIN_TIMEOUT_S} secondes.")
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        st.error(f"Erreur {exc.response.status_code} :\n\n{formater_erreur(detail)}")
    except httpx.HTTPError as exc:
        st.error(f"Erreur HTTP : {exc}")
    else:
        statut = "promu en production" if resultat_entrainement["promoted"] else "refusé"
        st.success(
            f"Réentraînement terminé : modèle {statut}. "
            f"{resultat_entrainement['training_rows']} ligne(s) utilisées, "
            f"dont {resultat_entrainement['feedback_rows_used']} correction(s) conseiller."
        )
        st.json(
            {
                "Statut": resultat_entrainement["status"],
                "Version du modèle": resultat_entrainement.get("artifact_version"),
                "Accuracy": resultat_entrainement["metrics"].get("accuracy"),
                "F1 macro": resultat_entrainement["metrics"].get("f1_macro"),
            }
        )

with st.form("formulaire_demandeur"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=63, value=30, step=1)
        anciennete_poste_ans = st.number_input(
            "Ancienneté au dernier poste (années)",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=0.5,
        )
        niveau_diplome = st.selectbox(
            "Niveau de diplôme",
            NIVEAUX_DIPLOME,
            index=1,
        )
        code_rome_vise = st.text_input(
            "Code ROME visé",
            value="",
            max_chars=5,
            placeholder="Ex : M1805",
        ).upper()
    with col2:
        code_insee_commune = st.text_input(
            "Code INSEE commune",
            value="",
            max_chars=5,
            placeholder="Ex : 75056",
        )
        est_allocataire = st.radio(
            "Allocataire",
            ["0", "1"],
            horizontal=True,
            format_func=lambda v: "Oui" if v == "1" else "Non",
        )
        nationalite_hors_ue = st.radio(
            "Nationalité hors UE",
            ["0", "1"],
            horizontal=True,
            format_func=lambda v: "Oui" if v == "1" else "Non",
        )

    synthese_entretien = st.text_area(
        "Synthèse de l'entretien",
        height=150,
        placeholder="Ex : Recherche active, mobile, formation en cours...",
    )

    valide = st.form_submit_button("Obtenir la prédiction", type="primary")

if valide:
    payload = {
        "age": int(age),
        "niveau_diplome": niveau_diplome,
        "anciennete_poste_ans": float(anciennete_poste_ans),
        "code_rome_vise": code_rome_vise,
        "code_insee_commune": code_insee_commune,
        "est_allocataire": est_allocataire,
        "nationalite_hors_ue": nationalite_hors_ue,
        "synthese_entretien": synthese_entretien,
    }
    try:
        with st.spinner("Prédiction en cours…"):
            headers = {"X-Conseiller-ID": conseiller_id} if conseiller_id else {}
            response = httpx.post(
                f"{API_URL}/predict",
                json=payload,
                headers=headers,
                timeout=TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        st.error("Erreur de connexion : impossible de joindre l'API.")
    except httpx.TimeoutException:
        st.error(f"API trop lente : pas de réponse en {TIMEOUT_S} secondes.")
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        st.error(f"Erreur {exc.response.status_code} :\n\n{formater_erreur(detail)}")
    except httpx.HTTPError as exc:
        st.error(f"Erreur HTTP : {exc}")
    else:
        retour_emploi = data["retour_emploi"]
        st.session_state.derniere_prediction = {
            "payload": payload,
            "classe_predite": retour_emploi,
        }
        st.session_state.feedback_enregistre = False
        afficher = COULEUR_CLASSE.get(retour_emploi, st.info)
        afficher(f"Retour à l'emploi estimé : **{retour_emploi}**")
        probabilites = data["probabilites"]
        df_probas = pd.DataFrame(
            {
                "classe": [c for c in ORDRE_CLASSES if c in probabilites],
                "probabilite": [probabilites[c] for c in ORDRE_CLASSES if c in probabilites],
            }
        )
        # st.bar_chart trierait les catégories par ordre alphabétique : on force l'ordre via altair.
        chart = (
            alt.Chart(df_probas)
            .mark_bar()
            .encode(
                x=alt.X("classe", sort=ORDRE_CLASSES, title=None),
                y=alt.Y("probabilite", title="Probabilité"),
            )
        )
        st.altair_chart(chart, use_container_width=True)

derniere_prediction = st.session_state.derniere_prediction
if derniere_prediction is not None:
    st.subheader("Corriger la prédiction")
    st.caption(f"Classe initialement prédite : {derniere_prediction['classe_predite']}")
    with st.form("formulaire_feedback"):
        classe_corrigee = st.selectbox(
            "Classe réellement constatée",
            ORDRE_CLASSES,
            index=ORDRE_CLASSES.index(derniere_prediction["classe_predite"]),
        )
        commentaire = st.text_area(
            "Commentaire facultatif",
            max_chars=500,
            placeholder="Précisions sur la correction...",
        )
        envoyer_feedback = st.form_submit_button(
            "Enregistrer la correction",
            disabled=st.session_state.feedback_enregistre,
        )

    if envoyer_feedback:
        feedback_payload = {
            **derniere_prediction["payload"],
            "classe_predite": derniere_prediction["classe_predite"],
            "classe_corrigee": classe_corrigee,
            "commentaire": commentaire or None,
        }
        try:
            with st.spinner("Enregistrement de la correction..."):
                reponse_feedback = httpx.post(
                    f"{API_URL}/feedback",
                    json=feedback_payload,
                    timeout=TIMEOUT_S,
                )
                reponse_feedback.raise_for_status()
                resultat_feedback = reponse_feedback.json()
        except httpx.ConnectError:
            st.error("Erreur de connexion : impossible d'enregistrer la correction.")
        except httpx.TimeoutException:
            st.error(f"API trop lente : pas de réponse en {TIMEOUT_S} secondes.")
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except ValueError:
                detail = exc.response.text
            st.error(f"Erreur {exc.response.status_code} :\n\n{formater_erreur(detail)}")
        except httpx.HTTPError as exc:
            st.error(f"Erreur HTTP : {exc}")
        else:
            st.session_state.feedback_enregistre = True
            st.success(
                "Correction enregistrée. "
                f"Total : {resultat_feedback['total_feedback_rows']} correction(s)."
            )

st.divider()
st.subheader("🕘 Historique des inférences")
col_bouton, col_limite = st.columns([1, 1])
with col_bouton:
    voir_historique = st.button("Charger l'historique", type="secondary")
with col_limite:
    limite_historique = st.number_input(
        "Nombre de lignes",
        min_value=5,
        max_value=200,
        value=50,
        step=5,
    )

if voir_historique:
    try:
        params = {"limit": int(limite_historique)}
        if conseiller_id:
            params["conseiller_id"] = conseiller_id
        with st.spinner("Chargement de l'historique…"):
            reponse_historique = httpx.get(f"{API_URL}/history", params=params, timeout=TIMEOUT_S)
            reponse_historique.raise_for_status()
            entries = reponse_historique.json().get("entries", [])
    except httpx.HTTPError as exc:
        st.error(f"Impossible de charger l'historique : {exc}")
    else:
        if not entries:
            st.info("Aucune prédiction enregistrée pour le moment.")
        else:
            st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)

with httpx.Client(base_url=API_URL, timeout=2) as client:
    try:
        health = client.get("/health").json()
        healthy = health.get("model_loaded")
        st.sidebar.success("✅ API joignable" if healthy else "⚠️ API en chargement")
        st.sidebar.markdown(f"**Version du modèle** : `{health.get('model_version', 'inconnue')}`")
    except httpx.HTTPError:
        st.sidebar.error("❌ API injoignable")

with st.sidebar:
    st.markdown(f"**API URL** : `{API_URL}`")
    st.markdown(
        "**Champs requis** :\n"
        "- Âge, ancienneté, diplôme\n"
        "- Code ROME visé, code INSEE\n"
        "- Allocataire, nationalité hors UE\n"
        "- Synthèse d'entretien"
    )
