# Pipeline de Ingesta & Streaming (`pipeline/`)

Directorio destinado para la configuración y persistencia del flujo de datos en tiempo real de ingeniería de datos.

- `checkpoints_kafka/`: Directorio autogenerado que almacena el estado interno de la *SparkSession*. Se encarga de guardar los *offsets* de lectura procesados para garantizar la tolerancia a fallos (*Fault Tolerance*) y una recuperación *exactly-once* al detener y reanudar el streaming desde Kafka.
