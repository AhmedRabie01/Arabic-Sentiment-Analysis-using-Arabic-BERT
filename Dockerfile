FROM python:3.10-slim

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Workdir
# -----------------------------
WORKDIR /app

# -----------------------------
# Python deps
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application
# -----------------------------
COPY . .

# -----------------------------
# Expose API
# -----------------------------
EXPOSE 8080

# -----------------------------
# Run API (inference only)
# -----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
