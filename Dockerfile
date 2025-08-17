FROM python:3.11-alpine

# Устанавливаем зависимости для сборки и запуска
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev

WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .
RUN mkdir -p logs && chmod +x main.py

# Порт для Coolify
EXPOSE 8080

# Команда запуска
CMD ["python", "main.py"]