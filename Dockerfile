FROM python:3.10.16-slim

# Install poetry without cache
RUN pip install --no-cache-dir poetry

# Set working directory
WORKDIR /app

# Copy only dependency files first for caching
COPY pyproject.toml poetry.lock ./

# Install dependencies without dev packages for smaller image
RUN poetry install --no-root --only main

# Copy the rest of the application
COPY . .

# Expose API port
EXPOSE 8080

# Set environment variables
ENV NAME=fin-rag \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
