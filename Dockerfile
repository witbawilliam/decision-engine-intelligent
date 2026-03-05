# --- Stage 1: The Builder ---
FROM python:3.11-slim AS builder

# Prevent python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Install minimal build tools for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv and upgrade pip early
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Install dependencies (utilizing Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: The Runtime ---
FROM python:3.11-slim AS runtime

# Set environment variables for the runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

WORKDIR /app

# Copy the pre-compiled virtual environment
COPY --from=builder /opt/venv /opt/venv

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
# We copy specific folders to avoid bringing in local garbage (even with .dockerignore)
COPY ./app ./app
COPY ./core ./core
COPY ./workers ./workers
COPY ./storage ./storage

# Security: Set up non-root user
RUN useradd -m mluser && chown -R mluser /app
USER mluser

# Expose FastAPI port
EXPOSE 8000

# Final entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]