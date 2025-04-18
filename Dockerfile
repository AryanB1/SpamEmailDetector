FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000
CMD ["python", "app/api.py"]