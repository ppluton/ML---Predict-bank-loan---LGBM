FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies (production only)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY api/ api/
COPY src/ src/
COPY artifacts/ artifacts/
COPY monitoring/ monitoring/
COPY main.py .

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
