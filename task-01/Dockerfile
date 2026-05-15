# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Dependency layer (cached unless pyproject.toml or uv.lock change) ---
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# --- Source layer ---
COPY src/ src/
COPY utils/ utils/
COPY main.py run_breakthrough.py ./

# Output and logs are written here; mount as volumes to persist on host.
VOLUME ["/app/output", "/app/logs"]

ENTRYPOINT ["uv", "run", "python"]
CMD ["main.py"]
