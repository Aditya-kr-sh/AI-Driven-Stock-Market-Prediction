FROM python:3.12-slim

# Install system compilation packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency configs
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree
COPY . .

# Install core ai_engine package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Launch FastAPI web application server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
