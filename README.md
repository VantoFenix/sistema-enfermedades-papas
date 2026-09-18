# Sistema Diagnóstico de Enfermedades Foliares mediante Percepción Computacional

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
Este proyecto implementa un sistema de visión por computadora en tiempo real capaz de detectar y diagnosticar enfermedades foliares en plantas (como antracnosis, roya o negrilla). Para aislar la lesión y evitar que el modelo se confunda con elementos del entorno (tierra, manos, otras ramas), el sistema utiliza una arquitectura de inferencia en dos etapas:
1. **Detección (YOLO):** Identifica y recorta (enmarca) la hoja objetivo dentro del fotograma de la cámara, aislándola del fondo complejo.
2. **Clasificación (ResNet18):** Analiza únicamente la región extraída de la hoja (Crop) procesando las características de textura y color para emitir el diagnóstico patológico preciso.

---

## 🗓️ Plan de Acción y Cronograma (Hitos)

El desarrollo está dividido en fases alineadas a los hitos de evaluación del curso de Percepción Computacional.

### Fase 1: Datos y Baseline Clásico (Hasta Semana 4 - Checkpoint)
* [ ] **Descarga y estructuración:** Obtención y balanceo de un dataset foliar (ej. PlantVillage o repositorio similar de Kaggle).
* [ ] **EDA (Exploratory Data Analysis):** Análisis de distribución de clases (hojas sanas vs. enfermas) y detección de anomalías de iluminación o fondo (01_eda.ipynb).
* [ ] **Preprocesamiento y Señales:** Extracción de descriptores clásicos, enfatizando LBP (Local Binary Patterns) para identificar las texturas de las enfermedades, y HOG para la morfología de la hoja (02_preprocesamiento_senales.ipynb).
* [ ] **Modelo Base:** Entrenamiento de un pipeline tradicional SVM + HOG/LBP como baseline riguroso y punto de comparación histórico (03_baseline_clasico.ipynb).

### Fase 2: Desarrollo del Pipeline de Dos Etapas (Hasta Semana 7 - Avance 1)
* [ ] **Etapa 1 (Detección):** Entrenar/ajustar modelo YOLO ligero para localizar la hoja en el entorno real y eliminar el sesgo visual del fondo.
* [ ] **Etapa 2 (Clasificación):** Entrenar arquitectura CNN (ej. ResNet18 o MobileNet) usando los recortes (crops) de las hojas para diagnosticar la patología específica.
* [ ] **Resultados:** Generar matrices de confusión y evaluar métricas robustas al desbalanceo (como el F1-Score) sobre el conjunto de prueba aislado.

### Fase 3: Integración y Despliegue MLOps (Hasta Semana 14 - Entrega Final)
* [ ] **Inferencia en vivo:** Integrar YOLO y ResNet con OpenCV para la captura de video y evaluación de muestras en tiempo real (inferencia_camara.py).
* [ ] **API:** Envolver el pipeline en un servicio REST usando FastAPI para recibir las imágenes y devolver el diagnóstico patológico.
* [ ] **Contenerización:** Crear Dockerfile garantizando la compatibilidad con el entorno de ejecución y las dependencias de visión.
* [ ] **Sustentación:** Grabación del video demo (5-8 min) evidenciando el sistema fito-sanitario funcionando de extremo a extremo (desde la cámara hasta la API).

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
