"""Middleware HTTP pour tracer les requêtes de l'API."""

import json
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-ID"


async def read_request_parameters(request: Request) -> dict | None:
    """Lit un JSON de requête sans exposer les champs textuels sensibles."""
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None

    body = await request.body()

    async def replay_body():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = replay_body
    try:
        parameters = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return {"json_invalide": True, "taille_octets": len(body)}

    if not isinstance(parameters, dict):
        return {"json_objet_attendu": True}

    text_value = parameters.pop("synthese_entretien", None)
    if text_value is not None:
        parameters["synthese_entretien_length"] = len(text_value)
    return parameters


async def read_response_body(response: Response) -> Response:
    """Récupère le contenu d'une réponse streaming sans le perdre."""
    body = b"".join([chunk async for chunk in response.body_iterator])
    return Response(
        content=body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


def response_detail(response: Response) -> object | None:
    """Extrait le détail JSON d'une réponse d'erreur FastAPI."""
    try:
        payload = json.loads(response.body or b"{}")
    except (TypeError, json.JSONDecodeError):
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list):
        return [
            {key: error[key] for key in ("type", "loc", "msg") if key in error}
            for error in detail
            if isinstance(error, dict)
        ]
    return detail


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Ajoute un identifiant de corrélation et journalise chaque requête."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        request.state.request_id = request_id
        started_at = perf_counter()

        with logger.contextualize(request_id=request_id):
            parameters = await read_request_parameters(request)
            if parameters is not None:
                logger.info(
                    "Requête reçue: {} {} | paramètres={}",
                    request.method,
                    request.url.path,
                    parameters,
                )
            else:
                logger.info("Requête reçue: {} {}", request.method, request.url.path)
            try:
                response = await call_next(request)
            except Exception:
                duration_ms = (perf_counter() - started_at) * 1000
                logger.exception(
                    "Requête en erreur: {} {} ({:.2f} ms)",
                    request.method,
                    request.url.path,
                    duration_ms,
                )
                raise

            if response.status_code >= 400:
                response = await read_response_body(response)
                detail = response_detail(response)
                if detail is not None:
                    logger.warning(
                        "Erreur HTTP: {} {} -> {} | détail={}",
                        request.method,
                        request.url.path,
                        response.status_code,
                        detail,
                    )

            duration_ms = (perf_counter() - started_at) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "Requête terminée: {} {} -> {} ({:.2f} ms)",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
