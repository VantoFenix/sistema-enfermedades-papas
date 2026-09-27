# Sistema de Diagnóstico de Enfermedades en Tubérculos de Papa mediante Percepción Computacional

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
| Velásquez Góngora, Bruno Martín | Ingeniería de datos (pipeline Spark/Kafka) · MLOps y documentación (Docker, MLflow, informe) |

---

## 📌 Resumen Ejecutivo
Este proyecto implementa un sistema de visión por computadora capaz de diagnosticar enfermedades en tubérculos de papa a partir de imágenes RGB. Se distinguen cuatro clases: **Blackspot Bruising**, **Dry Rot**, **Soft Rot** y **Healthy**. Para aislar el tubérculo y evitar que el modelo se confunda con el entorno (banda transportadora, fondo, manos), el sistema utiliza una arquitectura de inferencia en dos etapas:
1. **Detección (YOLO):** Localiza y recorta el tubérculo dentro del fotograma, aislándolo del fondo.
2. **Clasificación (ResNet18):** Analiza únicamente la región recortada, procesando las características de textura y color de la cáscara para emitir el diagnóstico.

Como punto de referencia se implementa un baseline clásico (descriptores HOG + histograma HSV con máscara Otsu y un clasificador estadístico).

---

## 🗓️ Plan de Acción y Cronograma (Hitos)

El desarrollo está dividido en fases alineadas a los hitos de evaluación del curso de Percepción Computacional.

### Fase 1: Datos y Baseline Clásico (Hasta Semana 4 - Checkpoint)
* [x] **Descarga y estructuración:** Dataset de tubérculos de papa (Mendeley Data), 3500 imágenes en 4 clases desbalanceadas. Ver [`data/README.md`](data/README.md).
* [x] **EDA:** Muestras por clase, histogramas de intensidad RGB, estadísticas de señal, detección de imágenes corruptas y duplicados (`01_eda.ipynb`).
* [x] **Partición:** Estratificada 70/15/15 con semilla fija 42 (`01_eda.ipynb`, `src/datos/particion.py`).
* [x] **Preprocesamiento y señales:** CLAHE en canal L (Lab), redimensionado a 224×224, análisis espectral FFT 2D, descriptor HOG e histogramas HSV con segmentación Otsu, aumento de datos solo en entrenamiento (`02_preprocesamiento_senales.ipynb`, `src/datos/`).
* [ ] **Modelo base:** Clasificador clásico sobre los descriptores HOG + HSV, optimizado con búsqueda de hiperparámetros sobre train (`03_baseline_clasico.ipynb`).

### Fase 2: Desarrollo del Pipeline de Dos Etapas (Hasta Semana 7 - Avance 1)
* [ ] **Etapa 1 (Detección):** Entrenar/ajustar un modelo YOLO ligero para localizar el tubérculo y eliminar el sesgo visual del fondo.
* [ ] **Etapa 2 (Clasificación):** Entrenar ResNet18 preentrenada en ImageNet sobre las imágenes de tubérculos para diagnosticar la enfermedad.
* [ ] **Resultados:** Matriz de confusión y métricas robustas al desbalance (F1 macro) sobre el conjunto de prueba aislado, comparando baseline vs. modelo avanzado.

### Fase 3: Integración y Despliegue MLOps (Hasta Semana 14 - Entrega Final)
* [ ] **Pipeline:** Integrar YOLO y ResNet18 en el pipeline de inferencia (`pipeline/streaming_job.py`).
* [ ] **API:** Servicio REST con FastAPI para recibir imágenes y devolver el diagnóstico (`servicio/app/main.py`).
* [ ] **Contenerización y versionado:** Dockerfile con versiones fijas y registro del modelo en MLflow.
* [ ] **Sustentación:** Video demo (5-8 min) del sistema funcionando de extremo a extremo.

---

## 📁 Estructura del Repositorio

```
sistema-enfermedades-papas/
├── README.md
├── requirements.txt           # versiones exactas
├── data/
│   └── README.md              # cómo obtener los datos (no se suben al repositorio)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocesamiento_senales.ipynb
│   └── 03_baseline_clasico.ipynb
├── src/
│   ├── datos/
│   │   ├── particion.py       # partición train/val/test
│   │   ├── features.py        # preprocesamiento + HOG + HSV/Otsu
│   │   └── transformaciones.py# aumento (train) y normalización (val/test)
│   ├── modelos/
│   ├── entrenar.py
│   └── evaluar.py
├── pipeline/
│   ├── streaming_job.py
│   └── README.md
├── servicio/
│   ├── Dockerfile
│   ├── app/main.py
│   └── requirements.txt
└── figuras/
```

---

## ⚙️ Reproducibilidad y Configuración del Entorno

### 1. Clonar el repositorio y crear el entorno virtual
```bash
git clone https://github.com/VantoFenix/sistema-enfermedades-papas.git
cd sistema-enfermedades-papas

python -m venv .venv
# En Windows: .\.venv\Scripts\activate
# En Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Obtener los datos
Seguir las instrucciones de [`data/README.md`](data/README.md).

### 3. Generar la partición (semilla 42)
```bash
python src/datos/particion.py --origen <ruta>/Potato_Dataset --destino <ruta>/Potato_Particionado --seed 42
```

### 4. Notebooks
Los notebooks se ejecutaron en Google Colab con el dataset montado desde Google Drive. Para reproducirlos, ajustar la variable `PAP` de la primera celda a la ruta local del dataset.

### 5. Entrenamiento y evaluación
*Pendiente (Fase 2).*
