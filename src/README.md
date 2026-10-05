# Código Fuente (`src/`)

Este directorio contiene todo el código productivo y de experimentación MLOps del proyecto.

## Archivos Principales

*   **`datos/dataset.py`**: Define la clase `PotatoDataset` para carga y transformación de tensores. Implementa de manera paramétrica la técnica de preprocesamiento CLAHE. Al final incluye un mini-script de validación rápida.
*   **`modelos/arquitectura.py`**: Contiene la definición de `DefectosPapaResNet`, una arquitectura basada en ResNet18 con *Transfer Learning* (congelación de capas) orientada a clasificar 4 defectos.
*   **`entrenar.py`**: Pipeline de entrenamiento avanzado. Implementa *Early Stopping*, pesos dinámicos por clase para contrarrestar el desbalance de los datos y guardado automático del mejor modelo (en `/models`).
*   **`evaluar.py`**: Script de inferencia que carga el modelo final, lo ejecuta sobre la partición estricta de *Test* (aislada) y exporta métricas y matrices de confusión.
*   **`evaluar_iluminacion.py`**: Script de *desempeño desagregado*. Evalúa el rendimiento particular del modelo dividiendo dinámicamente las imágenes según su brillo promedio (Bajo, Medio, Alto).
*   **`extraer_errores.py`**: Recorre las inferencias del modelo y aísla copias de casos de error muy específicos hacia la carpeta `/figuras` para su inspección cualitativa.
*   **`generar_graficos_informe.py`**: Automatiza la creación de visualizaciones académicas (Curvas de Aprendizaje simuladas, y mapas de calor explicativos con *Grad-CAM*).
*   **`latencia.py`**: Aísla el tiempo de inferencia crudo en milisegundos, comparando el motor *PyTorch* frente al pipeline clásico (*SVM + HOG*).
*   **`perfilado_escalamiento.py`**: Script de MLOps destinado a perfilar el cuello de botella entre el DataLoader (E/S en disco) y la iteración en GPU/CPU.
*   **`benchmark_escalabilidad.py`**: Simula una carga de alto volumen (5,000 imágenes) y mide el *speedup* real de migrar inferencias de Python Secuencial hacia PySpark distribuido.
*   **`pipeline_streaming.py`**: El código de Data Engineering final. Un Pipeline en *Apache Spark Structured Streaming* que captura video crudo desde *Kafka*, realiza predicciones masivas y devuelve las alertas al control industrial.
