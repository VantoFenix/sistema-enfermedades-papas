# Modelos Entrenados (`models/`)

Esta carpeta es el repositorio central donde se depositan los puntos de control (*checkpoints*) y pesos exportados durante los ciclos de entrenamiento.

- `defectos_v1.pt`: Archivo binario con los pesos estáticos (estado de diccionarios de PyTorch) del mejor modelo `DefectosPapaResNet` exportado tras aplicar *Early Stopping*. Este binario es levantado por el pipeline de Streaming para la inferencia en producción.
