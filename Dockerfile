FROM python:3.11-alpine

WORKDIR /app

# Устанавливаем зависимости для Alpine
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p logs && chmod +x main.py

EXPOSE 8080
CMD ["python", "main.py"]
