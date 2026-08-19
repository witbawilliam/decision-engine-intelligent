FROM ghcr.io/mlflow/mlflow:v2.13.0
RUN pip install psycopg2-binary --quiet
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
