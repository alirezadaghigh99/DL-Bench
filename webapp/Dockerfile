FROM python:3.11-slim

# Install Docker CLI so the app can spawn sibling per-repo containers via the
# host socket. (We mount /var/run/docker.sock at runtime — see compose file.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian bookworm stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/requirements.txt

COPY harness /app/harness
COPY webapp /app/webapp
COPY data /app/data

ENV DLBENCH_CSV=/app/data/DL-Bench-Enriched-Processed-Sorted.csv \
    DLBENCH_DB=/app/var/dlbench.sqlite \
    DLBENCH_REGISTRY=/app/harness/registry.yaml \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "600", \
     "webapp.app:create_app()"]
