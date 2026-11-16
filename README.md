# Sistema de Diagnóstico de Enfermedades en Tubérculos de Papa mediante Percepción Computacional ("HotPotato")

**Contexto:** Sistema de Percepción Computacional desarrollado para Veltri Software Solutions.  
**Institución:** Universidad Privada Antenor Orrego (UPAO) - Facultad de Ingeniería  
**Programa:** Escuela Profesional de Ingeniería de Sistemas e Inteligencia Artificial  
**Asignatura:** Percepción Computacional (ISIA-111) | Semestre: 2026-20  
**Docente:** Caballero Alvarado, Armando Javier | Grupo N.° 1  

### Equipo de Desarrollo
| Integrante | Rol |
|---|---|
| Gomez Salinas, Yessica | Datos y señales (adquisición, preprocesamiento, análisis espectral) |
| Trigoso Zarate, Tiago | Modelado (baseline, arquitectura, entrenamiento) |
| Velásquez Góngora, Bruno Martín | Ingeniería de datos (pipeline Spark/Kafka) · MLOps y despliegue (Docker, MLflow, FastAPI) |

---

## 📌 Resumen Ejecutivo
Este proyecto implementa un sistema integral de visión por computadora y MLOps capaz de clasificar y diagnosticar patologías en tubérculos de papa a partir de imágenes RGB. Se distinguen cuatro clases diagnósticas:
* **Blackspot Bruising** (Manchas negras por golpe o presión)
* **Dry Rot** (Fusariosis o pudrición seca)
* **Healthy Potatoes** (Tubérculos sanos)
* **Soft Rot** (Pudrición blanda bacteriana)

El núcleo predictivo utiliza una red neuronal convolucional profunda **ResNet18** ajustada mediante *Transfer Learning* (congelación de bloques 1 a 3 y reentrenamiento del bloque 4 y cabezal clasificador), complementada con un pipeline de streaming distribuido en **Apache Spark Structured Streaming** y **Apache Kafka**, y un microservicio REST productivo contenerizado con **FastAPI** y **Docker**.

---

## 🗓️ Plan de Acción y Cronograma (Hitos Cumplidos)

### Fase 1: Datos y Baseline Clásico (Completado)
* [x] **Descarga y estructuración:** Dataset de tubérculos de papa (Mendeley Data), 3500 imágenes en 4 clases desbalanceadas. Ver [`data/README.md`](data/README.md).
* [x] **EDA:** Muestras por clase, histogramas de intensidad RGB, estadísticas de señal, detección de imágenes corruptas y duplicados (`01_eda.ipynb`).
* [x] **Partición:** Estratificada 70/15/15 con semilla fija 42 (`01_eda.ipynb`, `src/datos/particion.py`).
* [x] **Preprocesamiento y señales:** CLAHE en canal L (Lab), redimensionado a 224×224, análisis espectral FFT 2D, descriptor HOG e histogramas HSV con segmentación Otsu (`02_preprocesamiento_senales.ipynb`, `src/datos/`).
* [x] **Modelo base:** Clasificador clásico (SVM con kernel RBF) sobre descriptores HOG + HSV (`03_baseline_clasico.ipynb`).

### Fase 2: Modelado Profundo y Experiment Tracking (Completado)
* [x] **Arquitectura:** Red `DefectosPapaResNet` sobre ResNet18 con pesos ImageNet y cabezal personalizado (`src/modelos/arquitectura.py`).
* [x] **Entrenamiento Avanzado:** Manejo de desbalance mediante pérdida con pesos de clase inversos y *Early Stopping* (`src/entrenar.py`).
* [x] **Experiment Tracking con MLflow:** Matriz de 4 corridas comparativas con registro de métricas por época, artefactos y publicación en el **Model Registry** (`src/entrenar_mlflow.py`).

### Fase 3: Despliegue, Inferencia y Streaming MLOps (Completado)
* [x] **Pipeline de Streaming Distribuido:** Inferencia en micro-lotes con Spark Structured Streaming, conector Kafka y Pandas UDF vectorizada (`src/pipeline_streaming.py`, `pipeline/streaming_job.py`).
* [x] **API REST Productiva:** Microservicio FastAPI con endpoints `/health` y `/predict`, carga Singleton del modelo en memoria (`servicio/main.py`).
* [x] **Benchmarking de Latencia:** Script de evaluación empírica con 150 llamadas HTTP consecutivas, validando percentil 95 (P95 < 150 ms) (`servicio/benchmark.py`).
* [x] **Contenerización Docker:** Imagen optimizada basada en `python:3.11-slim` con PyTorch CPU, sin sobrepeso de dependencias y con `HEALTHCHECK` (`Dockerfile`).

---

## 📁 Estructura del Repositorio

```
sistema-enfermedades-papas/
├── Dockerfile                 # Contenedor de producción (python:3.11-slim)
├── requirements.txt           # Dependencias fijadas estrictamente
├── README.md                  # Guía de reproducción y documentación técnica
├── data/
│   ├── README.md              # Origen del dataset y cómo estructurarlo
│   ├── Potato_Dataset/        # Datos crudos organizados por clase
│   └── Potato_Particionado/   # Partición determinista (train/val/test)
├── notebooks/
│   ├── 01_eda.ipynb           # Análisis exploratorio de datos
│   ├── 02_preprocesamiento_senales.ipynb # CLAHE, FFT 2D y análisis de color
│   └── 03_baseline_clasico.ipynb # Modelo clásico SVM + HOG/HSV
├── src/
│   ├── datos/
│   │   ├── dataset.py         # DataLoader PyTorch con soporte CLAHE
│   │   ├── particion.py       # Particionado estratificado 70/15/15
│   │   ├── features.py        # Extracción de características
│   │   └── transformaciones.py# Aumentación y normalización
│   ├── modelos/
│   │   ├── arquitectura.py    # Definición de DefectosPapaResNet (ResNet18)
│   │   └── evaluar.py         # Evaluación de modelos
│   ├── entrenar.py            # Entrenamiento estándar con early stopping
│   ├── entrenar_mlflow.py     # Experiment tracking y Model Registry con MLflow
│   ├── evaluar.py             # Matriz de confusión y métricas en Test
│   ├── latencia.py            # Comparativa de latencia PyTorch vs SVM
│   └── pipeline_streaming.py  # Spark Structured Streaming + Kafka (Vectorizado)
├── pipeline/
│   ├── streaming_job.py       # Réplica del pipeline para ejecución distribuida
│   └── checkpoints_kafka/     # Checkpoint para tolerancia a fallos
├── servicio/
│   ├── main.py                # API REST con FastAPI (GET /health, POST /predict)
│   ├── benchmark.py           # Benchmark de latencia real (150 llamadas HTTP, P95)
│   ├── requirements.txt       # Dependencias mínimas del microservicio
│   └── app/
│       └── main.py            # Re-export para retrocompatibilidad
├── models/
│   ├── README.md              # Documentación de pesos y checkpoints
│   └── defectos_v1.pt         # Pesos óptimos del modelo ganador
├── frontend/                  # Dashboard React + Vite para inspección en vivo
│   ├── src/                   # Componentes y estilos industriales
│   ├── package.json           # Dependencias del cliente web
│   └── vite.config.js         # Configuración del servidor Vite
└── figuras/                   # Matrices de confusión y gráficos generados
```

---

## ⚙️ Reproducibilidad y Configuración del Entorno

### 1. Clonar el repositorio y crear el entorno virtual

```bash
git clone https://github.com/VantoFenix/sistema-enfermedades-papas.git
cd sistema-enfermedades-papas

# Crear entorno virtual con Python 3.11
python -m venv .venv

# Activar entorno virtual
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# En Linux/macOS:
source .venv/bin/activate
```

### 2. Instalar dependencias fijadas estrictamente

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar las versiones exactas sin inflar dependencias
pip install -r requirements.txt
```

### 3. Obtener los datos y generar la partición estratificada (Semilla 42)

Descargar el dataset siguiendo las instrucciones en [`data/README.md`](data/README.md) y colocarlo en `data/Potato_Dataset`. Luego, generar la división determinista 70/15/15:

```bash
python src/datos/particion.py --origen data/Potato_Dataset --destino data/Potato_Particionado --seed 42
```

### 4. Entrenamiento y Experiment Tracking con MLflow

El script ejecuta 4 corridas comparativas de hiperparámetros (variando tasa de aprendizaje, optimizador `AdamW` vs `SGD` y ablación de CLAHE). Guarda los pesos de cada corrida de forma aislada, selecciona automáticamente el modelo con mayor $F_1$-Macro en validación, copia sus pesos a `models/defectos_v1.pt` y lo registra en el **Model Registry** de MLflow:

```bash
python src/entrenar_mlflow.py
```

Para visualizar las curvas de aprendizaje, parámetros y modelos registrados en la interfaz web de MLflow:

```bash
mlflow ui --port 5000
```
Abrir en el navegador: [http://localhost:5000](http://localhost:5000)

### 5. Despliegue Local de la API REST con FastAPI

Iniciar el servidor Uvicorn en local:

```bash
uvicorn servicio.main:app --host 0.0.0.0 --port 8000 --reload
```

* **Documentación interactiva Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Verificación de estado (Healthcheck):**
  ```bash
  curl -X GET http://localhost:8000/health
  # Respuesta esperada: {"status":"ok","model":"HotPotato-ResNet18","version":"1"}
  ```
* **Predicción manual vía cURL:**
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@data/Potato_Particionado/test/Healthy Potatoes/healthy_sample.jpg"
  ```

### 6. Medición de Latencia Real en Producción

Con el servidor API en ejecución, abrir otra terminal y ejecutar el benchmark de 150 llamadas HTTP consecutivas:

```bash
python servicio/benchmark.py
```

El script reporta:
* Total de llamadas exitosas
* Latencia mínima, media, mediana (**P50**) y percentil 95 (**P95**)
* Verificación del umbral de SLA industrial ($P95 < 150 \text{ ms}$)

### 7. Despliegue con Docker (Contenerización)

Construir la imagen Docker ligera (`python:3.11-slim` con PyTorch CPU):

```bash
# Construir la imagen
docker build -t hotpotato-api:1.0.0 .

# Ejecutar el contenedor exponiendo el puerto 8000
docker run -d --name hotpotato-servicio -p 8000:8000 hotpotato-api:1.0.0

# Verificar el estado del HEALTHCHECK en el contenedor
docker inspect --format='{{json .State.Health.Status}}' hotpotato-servicio

# Probar la API dentro del contenedor
curl http://localhost:8000/health
```

Para detener el contenedor:
```bash
docker stop hotpotato-servicio
docker rm hotpotato-servicio
```

### 8. Frontend Web de Inspección en Línea (React + Vite)

Para iniciar la interfaz interactiva de control de calidad:

```bash
cd frontend

# Instalar dependencias del cliente web
npm install

# Iniciar servidor de desarrollo en http://localhost:5173
npm run dev
```

La interfaz permite cargar imágenes, verificar el healthcheck en vivo contra `http://localhost:8000/health` y visualizar las decisiones automáticas de compuerta/eyector mecánico con desglose de probabilidades y medición de latencia E2E.

### 9. Pipeline Distribuido de Streaming (Spark + Kafka)

Para procesar flujos masivos de video en tiempo real desde la cinta transportadora:

```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  --master local[*] \
  src/pipeline_streaming.py
```
* **Tópico de entrada (Kafka):** `raw-video-stream`
* **Tópico de salida (Alertas de eyección):** `alertas-eyeccion`
* **Checkpoints:** `pipeline/checkpoints_kafka/`
