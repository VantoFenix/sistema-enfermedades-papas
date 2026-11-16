# 1. Imagen base oficial con versión estricta fija (no latest)
FROM python:3.11-slim

# Metadatos del contenedor
LABEL maintainer="Grupo 1 - Percepción Computacional"
LABEL description="Microservicio de clasificación de defectos en tubérculos de papa (HotPotato)"

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Instalar curl para HEALTHCHECK y dependencias mínimas de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt a la raíz de trabajo
COPY requirements.txt .

# Instalar PyTorch CPU ligero primero (para ahorrar >3.5GB de binarios CUDA) y dependencias fijadas
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copiar servicio, código fuente y pesos entrenados del modelo
COPY servicio/ /app/servicio/
COPY src/ /app/src/
COPY models/defectos_v1.pt /app/models/defectos_v1.pt

# Exponer el puerto de FastAPI
EXPOSE 8000

# Verificación de salud periódica mediante healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de arranque del servidor ASGI
CMD ["uvicorn", "servicio.main:app", "--host", "0.0.0.0", "--port", "8000"]
