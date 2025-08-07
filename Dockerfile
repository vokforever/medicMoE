# Этап 1: Установка зависимостей
FROM python:3.11-alpine as builder

# Устанавливаем только необходимые зависимости для сборки
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev

WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости с оптимизациями
RUN pip install --no-cache-dir --user --prefix=/install \
    --no-compile \
    -r requirements.txt

# Этап 2: Финальный образ
FROM python:3.11-alpine

# Устанавливаем только runtime зависимости
RUN apk add --no-cache libffi libssl

WORKDIR /app

# Копируем установленные зависимости из builder
COPY --from=builder /install /usr/local

# Копируем код приложения
COPY . .

# Создаем директорию для логов и устанавливаем права
RUN mkdir -p logs && chmod +x main.py

# Порт для Coolify
EXPOSE 8080

# Команда запуска
CMD ["python", "main.py"]
