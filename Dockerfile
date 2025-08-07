FROM python:3.11-slim as builder

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Настройка окружения
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Финальный образ
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /home/app/.local
COPY . .
RUN mkdir -p logs && chmod +x main.py
EXPOSE 8080
CMD ["python", "main.py"]
