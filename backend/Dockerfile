FROM python:3.11-slim

WORKDIR /app

# Layer 1: System packages (cached)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python requirements (only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: Application source code (rebuilds quickly on code commits)
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]