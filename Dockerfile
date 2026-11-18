FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    EXPERIMENTS_DIR=/models/experiments \
    MODELS_DIR=/models/artifacts \
    HF_HOME=/cache/huggingface

COPY requirements.txt .

# CPU wheels only. The checkpoints are mounted later; they are not copied here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir hazm --no-deps

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
