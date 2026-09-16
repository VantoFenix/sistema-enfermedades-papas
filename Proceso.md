# Hoja de Ruta de Desarrollo (Playbook)

Este documento detalla el paso a paso técnico para poblar cada carpeta del proyecto, entrenar los modelos y construir el pipeline de inferencia final.

## PASO 1: Preparación del Dataset (`data/` y `src/datos/`)
1. **Descargar Mendeley:** Bajar el dataset de Mendeley (tubérculos sanos y enfermos) y colocarlo en `data/raw/mendeley`.
2. **Conseguir Dataset para YOLO:** Como Mendeley no tiene coordenadas, descargar un dataset secundario genérico (ej. de Roboflow) que solo detecte "Papas". Guardarlo en `data/raw/yolo_papas`.
3. **Scripts de Partición:** En `src/datos/`, crear un script que tome las imágenes crudas, las redimensione y las separe en 70% entrenamiento, 15% validación y 15% prueba con una semilla fija (`seed=42`). Guardar el resultado en `data/processed/`.

## PASO 2: Análisis y Baseline Clásico (`notebooks/`)
*Esta fase es obligatoria para cumplir con la rúbrica de la universidad.*
1. **01_eda.ipynb:** Cargar las imágenes procesadas. Graficar la distribución de clases (cuántas imágenes sanas vs enfermas) y mostrar ejemplos visuales de cada una.
2. **02_preprocesamiento_senales.ipynb:** Aplicar filtros (ej. filtro Gaussiano para ruido) y extraer descriptores clásicos como HOG (para los bordes/forma) y LBP (para la textura de la cáscara). 
3. **03_baseline_clasico.ipynb:** Entrenar un modelo tradicional (SVM o Random Forest) usando los descriptores HOG/LBP extraídos. Medir la exactitud inicial como punto de referencia.

## PASO 3: Etapa 1 - Modelo de Detección (`src/modelos/etapa1_yolo/`)
1. **Configuración:** Instalar la librería `ultralytics`.
2. **Entrenamiento:** Crear el script `train_yolo.py`. Configurar el entrenamiento para usar la aceleración CUDA aprovechando la GPU RTX 4050 local. Esto reducirá el tiempo de entrenamiento de días a horas o minutos.
3. **Objetivo:** Entrenar a YOLOv8/11 únicamente para encontrar dónde está la papa en una imagen. Guardar los pesos finales (`best.pt`) en esta misma carpeta.

## PASO 4: Etapa 2 - Modelo de Clasificación (`src/modelos/etapa2_resnet/`)
1. **Arquitectura:** Usar `PyTorch` para importar un modelo `ResNet18` preentrenado.
2. **Entrenamiento:** Crear `train_resnet.py`. Entrenarlo exclusivamente con el dataset procesado de Mendeley. 
3. **Objetivo:** Que el modelo reciba una imagen cuadrada de una papa y devuelva la clase exacta (Sarna Común, Podredumbre Seca, Sana, etc.). Guardar los pesos finales (`resnet_best.pth`).

## PASO 5: El Pipeline en Vivo (`src/pipeline/`)
1. **Script Principal:** En `inferencia_camara.py`, importar OpenCV (`cv2`).
2. **Flujo de Ejecución:**
   * Abrir la cámara web `cap = cv2.VideoCapture(0)`.
   * En cada fotograma, pasar la imagen a **YOLO**.
   * Si YOLO detecta una papa, recortar (crop) matemáticamente ese recuadro.
   * Pasar ese recorte a **ResNet18** para que dé el diagnóstico.
   * Dibujar el recuadro original en pantalla y poner el texto del diagnóstico encima.
   * Mostrar el video en pantalla en tiempo real.

## PASO 6: Despliegue y MLOps (`src/servicio/`)
1. **API:** En `app/main.py`, construir una API con FastAPI que tenga un endpoint `/predict`. Debe recibir una imagen en base64 y devolver el JSON con el diagnóstico.
2. **Contenedor:** Escribir el `Dockerfile` instalando las versiones exactas (`requirements.txt`) para empaquetar el servicio y dejarlo listo para producción.