# Figuras Exportadas (`figuras/`)

Esta carpeta almacena todos los *outputs* gráficos generados por los scripts del proyecto, preparados para ser inyectados directamente en los informes:

- `curvas_entrenamiento.png`: Visualización académica de *Loss* y *Accuracy* (Entrenamiento vs. Validación) por cada época.
- `gradcam.png`: Mapa de calor de interpretabilidad visual resaltando en qué regiones morfológicas se basó la *ResNet18* para su predicción.
- `matriz_confusion_test.png`: Matriz de confusión normalizada calculada estrictamente en el conjunto de prueba de forma imparcial.
- `error_*.png`: Muestras reales extraídas programáticamente de las equivocaciones específicas del modelo para análisis de debilidades.
