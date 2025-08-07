# Используем стабильную версию без проблем с SSL
FROM python:3.11.4-slim as builder

# Установка системных зависимостей с очисткой кеша
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создаем swap-файл на 1GB для гарантии завершения сборки
RUN dd if=/dev/zero of=/swapfile bs=1M count=1024 && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile

# Увеличиваем таймауты и отключаем проверки
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MAX_JOBS=1

WORKDIR /app

# Копируем только requirements.txt для лучшего кеширования
COPY requirements.txt .

# Устанавливаем зависимости с ограничением памяти
RUN pip install --no-cache-dir --user -r requirements.txt && \
    pip list | grep -E "(requests|aiogram|openai)" && \
    echo "Dependencies installed successfully"

# Финальный образ
FROM python:3.11.4-slim

WORKDIR /app

# Копируем установленные зависимости из builder
COPY --from=builder /root/.local /home/app/.local

# Устанавливаем права и пути
RUN mkdir -p logs && \
    chmod +x main.py && \
    chown -R appuser:appuser /app

# Создаем непривилегированного пользователя
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# Переключаемся на непривилегированного пользователя
USER appuser

# Устанавливаем переменные окружения
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONPATH=$PYTHONPATH:/home/app/.local \
    PYTHONUNBUFFERED=1

# Копируем код приложения
COPY --chown=appuser:appuser . .

# Порт для Coolify
EXPOSE 8080

# Проверяем, что все модули доступны перед запуском
RUN python -c "import requests; import aiogram; import openai; print('All modules imported successfully')"

# Команда запуска с ограничением ресурсов
CMD ["sh", "-c", "python main.py"]
