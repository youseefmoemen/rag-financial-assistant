FROM python:3.10.16-slim

RUN pip install --no-cache-dir poetry

RUN mkdir app

WORKDIR /app


COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root



COPY . .


EXPOSE 8080


ENV NAME=fin-rag

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]
