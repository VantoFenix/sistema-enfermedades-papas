# Sistema Diagnóstico de Enfermedades en Tubérculos de Papa con Percepción Computacional

**Contexto:** Sistema de Percepción Computacional desarrollado para Veltri Software Solutions.
**Institución:** Universidad Privada Antenor Orrego (UPAO) - Facultad de Ingeniería
**Programa:** Escuela Profesional de Ingeniería de Sistemas e Inteligencia Artificial
**Asignatura:** Percepción Computacional (ISIA-111) | Semestre: 2026-20
**Docente:** Caballero Alvarado, Armando Javier | Grupo N.° 1

### Equipo de Desarrollo
* Gomez Salinas, Yessica
* Trigoso Zarate, Tiago
* Velásquez Góngora, Bruno Martín

---

## 📌 Resumen Ejecutivo
Este proyecto implementa un sistema de visión por computadora en tiempo real capaz de detectar y diagnosticar enfermedades superficiales en tubérculos de papa (como la Sarna Común y Podredumbre Seca). Para superar la limitación del dataset base (Mendeley) que carece de anotaciones de coordenadas, el sistema utiliza una arquitectura de inferencia en dos etapas:
1. **Detección (YOLO):** Identifica y enmarca el tubérculo en el fotograma de la cámara.
2. **Clasificación (ResNet18):** Analiza la región extraída (Crop) para emitir el diagnóstico de la enfermedad.

---

## 🗓️ Plan de Acción y Cronograma (Hitos)

El desarrollo está dividido en fases alineadas a los hitos de evaluación del curso.

### Fase 1: Datos y Baseline Clásico (Hasta Semana 4 - Checkpoint)
* [ ] **Descarga y estructuración:** Obtención del dataset de Mendeley.
* [ ] **EDA (Exploratory Data Analysis):** Análisis de distribución de clases y detección de anomalías (`01_eda.ipynb`).
* [ ] **Preprocesamiento y Señales:** Extracción de descriptores clásicos como HOG y LBP (`02_preprocesamiento_senales.ipynb`).
* [ ] **Modelo Base:** Entrenamiento de un pipeline tradicional SVM + HOG como punto de comparación (`03_baseline_clasico.ipynb`).

### Fase 2: Desarrollo del Pipeline de Dos Etapas (Hasta Semana 7 - Avance 1)
* [ ] **Etapa 1 (Detección):** Entrenar/ajustar modelo YOLO ligero para detectar la papa en el entorno.
* [ ] **Etapa 2 (Clasificación):** Entrenar arquitectura CNN (ResNet18) usando el dataset de Mendeley procesado.
* [ ] **Resultados:** Generar matrices de confusión y análisis de errores sobre el conjunto de prueba aislado.

### Fase 3: Integración y Despliegue MLOps (Hasta Semana 14 - Entrega Final)
* [ ] **Inferencia en vivo:** Integrar YOLO y ResNet con OpenCV para captura en tiempo real (`inferencia_camara.py`).
* [ ] **API:** Envolver el pipeline en un servicio REST usando FastAPI.
* [ ] **Contenerización:** Crear `Dockerfile` garantizando compatibilidad con el entorno de ejecución.
* [ ] **Sustentación:** Grabación del video demo (5-8 min) con el sistema funcionando de extremo a extremo.

---

## ⚙️ Reproducibilidad y Configuración del Entorno

Para ejecutar este proyecto de forma local aprovechando la aceleración por hardware (NVIDIA RTX 4050 / CUDA), siga estos pasos:

2. Crear entorno virtual e instalar dependencias
Bash
python -m venv .venv
# En Windows: .\.venv\Scripts\activate
# En Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
3. Ejecución de la Interfaz de Cámara
Una vez descargados los pesos de los modelos en la carpeta models/, inicie el script principal:

Bash
python src/pipeline/inferencia_camara.py