# Build stage
FROM python:3.10.10-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.10.10-slim

COPY --from=builder /root/.local /root/.local

WORKDIR /app

COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
