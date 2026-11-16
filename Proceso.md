# Hoja de Ruta de Desarrollo (Playbook)

Este documento detalla el paso a paso técnico para poblar cada carpeta del proyecto, entrenar los modelos y construir el pipeline de inferencia final enfocado en el diagnóstico foliar (hojas).

## PASO 1: Preparación del Dataset (`data/` y `src/datos/`)
1. **Descargar PlantVillage:** Bajar el dataset de PlantVillage (sección de hojas de papa: sanas, con Tizón Temprano y Tizón Tardío) y colocarlo en `data/raw/plantvillage`.
2. **Conseguir Dataset para YOLO:** Como PlantVillage no tiene coordenadas, descargar un dataset secundario genérico (ej. de Roboflow) enfocado en detectar "Hojas de plantas". Guardarlo en `data/raw/yolo_hojas`.
3. **Scripts de Partición:** En `src/datos/`, crear un script que tome las imágenes crudas, las redimensione y las separe en 70% entrenamiento, 15% validación y 15% prueba con una semilla fija (`seed=42`). Guardar el resultado en `data/processed/`.

## PASO 2: Análisis y Baseline Clásico (`notebooks/`)
*Esta fase es obligatoria para cumplir con la rúbrica de la universidad.*
1. **01_eda.ipynb:** Cargar las imágenes procesadas. Graficar la distribución de clases (cuántas imágenes sanas vs con Tizón) y mostrar ejemplos visuales de cada categoría.
2. **02_preprocesamiento_senales.ipynb:** Aplicar filtros (ej. filtro Gaussiano para ruido). Dado que en las hojas el color es un factor diagnóstico crítico, extraer descriptores basados en el color (como Histogramas en el espacio HSV para capturar zonas de clorosis o necrosis) combinados con HOG (para los bordes y estructura de las venas de la hoja). 
3. **03_baseline_clasico.ipynb:** Entrenar un modelo tradicional (SVM o Random Forest) usando los descriptores HSV/HOG extraídos. Medir la exactitud inicial como punto de referencia.

## PASO 3: Etapa 1 - Modelo de Detección (`src/modelos/etapa1_yolo/`)
1. **Configuración:** Instalar la librería `ultralytics`.
2. **Entrenamiento:** Crear el script `train_yolo.py`. Configurar el entrenamiento para usar la aceleración CUDA aprovechando la GPU local para reducir drásticamente los tiempos de procesamiento.
3. **Objetivo:** Entrenar a YOLOv8/11 únicamente para encontrar dónde está la hoja en el entorno de la cámara. Guardar los pesos finales (`best.pt`) en esta misma carpeta.

## PASO 4: Etapa 2 - Modelo de Clasificación (`src/modelos/etapa2_resnet/`)
1. **Arquitectura:** Usar `PyTorch` para importar un modelo `ResNet18` preentrenado.
2. **Entrenamiento:** Crear `train_resnet.py`. Entrenarlo exclusivamente con el dataset procesado de PlantVillage. 
3. **Objetivo:** Que el modelo reciba una imagen cuadrada de una hoja y devuelva la clase exacta (Tizón Temprano, Tizón Tardío o Sana). Guardar los pesos finales (`resnet_best.pth`).

## PASO 5: El Pipeline en Vivo (`src/pipeline/`)
1. **Script Principal:** En `inferencia_camara.py`, importar OpenCV (`cv2`).
2. **Flujo de Ejecución:**
   * Abrir la cámara web `cap = cv2.VideoCapture(0)`.
   * En cada fotograma, pasar la imagen a **YOLO**.
   * Si YOLO detecta una hoja, recortar (crop) matemáticamente ese recuadro.
   * Pasar ese recorte a **ResNet18** para que emita el diagnóstico foliar.
   * Dibujar el recuadro original en pantalla y poner el texto del diagnóstico encima.
   * Mostrar el video en pantalla en tiempo real.

## PASO 6: Despliegue y MLOps (`src/servicio/`)
1. **API:** En `app/main.py`, construir una API con FastAPI que tenga un endpoint `/predict`. Debe recibir una imagen en base64 y devolver el JSON con el diagnóstico de la hoja.
2. **Contenedor:** Escribir el `Dockerfile` instalando las versiones exactas (`requirements.txt`) para empaquetar el servicio y dejarlo listo para producción.