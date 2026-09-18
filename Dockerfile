FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# libgomp1 : runtime OpenMP requis par lightgbm au chargement du modèle
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ./app/requirements.txt
RUN pip install --upgrade pip && pip install -r app/requirements.txt

COPY app/ ./app/

# models/, data/ et logs/ sont montés en volumes (docker-compose) : le chargement
# fail-fast du lifespan échoue si l'artefact modèle n'est pas présent au démarrage.
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
