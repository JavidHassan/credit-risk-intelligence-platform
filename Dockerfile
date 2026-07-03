FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Generate data and train model
RUN python -m src.data_generation.generate_synthetic_data || true

# Expose API and Dashboard ports
EXPOSE 8000 8501

# Default: run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
