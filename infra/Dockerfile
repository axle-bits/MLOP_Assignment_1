FROM python:3.13-slim

WORKDIR /app

# Dependencies first: this layer only rebuilds when pins change.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# The pickle resolves ml.features.clinical.add_clinical_features by import,
# so the ml package ships alongside the model artifact.
COPY ml/ /app/ml/
COPY models/ /app/models/
COPY api/ /app/api/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
