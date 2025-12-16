FROM python:3.12.3-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /midterm

ENV PATH="/midterm/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" "./"

RUN uv sync --locked

COPY "predict.py" "heartfailure_model.bin" "./"

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]