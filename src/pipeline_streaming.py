import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, from_json
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, FloatType
import pandas as pd
import torch
import numpy as np

# Asegurar que el path de src esté disponible para que el worker encuentre el modelo
# Nota: En un entorno distribuido real, se debe enviar este path con --py-files
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet

# ---------------------------------------------------------
# 1. Patrón Singleton para el Modelo PyTorch en Ejecutores
# ---------------------------------------------------------
# Se inicializa vacío a nivel de módulo. 
# Esto garantiza que cada ejecutor (Worker) cargue el modelo UNA SOLA VEZ 
# y no una vez por cada fila procesada, mitigando severos cuellos de botella.
_modelo = None

def get_model():
    """
    Función Singleton que inicializa los pesos de ResNet18 solo si no existen
    previamente en la memoria de la partición actual.
    """
    global _modelo
    if _modelo is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(project_root, 'models', 'defectos_v1.pt')
        
        _modelo = DefectosPapaResNet(num_classes=4)
        # Solo inferencia (eval)
        _modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _modelo = _modelo.to(device)
        _modelo.eval()
    return _modelo

# ---------------------------------------------------------
# 2. Inferencia Distribuida con Pandas UDF
# ---------------------------------------------------------
# Permite usar la API de DataFrames (más rápida por Apache Arrow) en lugar de RDDs.
@pandas_udf(StringType(), PandasUDFType.SCALAR)
def predict_defect_udf(tensor_series: pd.Series) -> pd.Series:
    """
    Recibe un lote (Batch) de tensores aplanados de Kafka, pasa por la ResNet18
    y retorna la clase identificada.
    """
    modelo = get_model()
    device = next(modelo.parameters()).device
    
    predictions = []
    
    with torch.no_grad():
        for flat_tensor in tensor_series:
            if flat_tensor is None:
                predictions.append("Error_Kafka")
                continue
                
            try:
                # Reconstruimos la imagen desde el JSON (1 dimensión -> 3x224x224)
                np_array = np.array(flat_tensor, dtype=np.float32).reshape(3, 224, 224)
                input_tensor = torch.tensor(np_array).unsqueeze(0).to(device)
                
                # Forward Pass
                output = modelo(input_tensor)
                _, pred = torch.max(output, 1)
                
                clases = ['Blackspot Bruising Care', 'Dry Rot Care', 'Healthy Potatoes', 'Soft Rot Care']
                predictions.append(clases[pred.item()])
                
            except Exception as e:
                # Tolerancia a Fallos: Evitar que el Pipeline colapse por un frame corrupto
                predictions.append("Frame_Corrupto")
                
    return pd.Series(predictions)


def main():
    # ---------------------------------------------------------
    # 3. Inicialización de SparkSession
    # ---------------------------------------------------------
    # Para correr este script en consola se requiere inyectar el conector de Kafka:
    # spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 src/pipeline_streaming.py
    spark = SparkSession.builder \
        .appName("Pipeline_Ejeccion_Papas_Streaming") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    print("\n--- Iniciando Spark Structured Streaming (Inferencia PyTorch) ---")

    # ---------------------------------------------------------
    # 4. Lectura de Stream (Kafka Source)
    # ---------------------------------------------------------
    # Simulación de un stream de video en crudo (cámaras de la cinta transportadora)
    # Se limita maxOffsetsPerTrigger para controlar Backpressure y evitar OOM.
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "raw-video-stream") \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 50) \
        .load()
        
    # Parseamos el JSON que viene como array de bytes binarios desde Kafka
    esquema = StructType([
        StructField("frame_id", StringType(), True),
        StructField("tensor", ArrayType(FloatType()), True)
    ])
    
    parsed_df = kafka_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), esquema).alias("data")) \
        .select("data.*")
        
    # ---------------------------------------------------------
    # 5. Procesamiento (Aplicar UDF)
    # ---------------------------------------------------------
    # Generamos la nueva columna con el veredicto de la IA
    predictions_df = parsed_df.withColumn(
        "resultado_ia",
        predict_defect_udf(col("tensor"))
    )
    
    # Preparamos el payload de salida a Kafka (ID del frame + Resultado)
    output_df = predictions_df.selectExpr(
        "frame_id AS key",
        "to_json(struct(frame_id, resultado_ia)) AS value"
    )

    # ---------------------------------------------------------
    # 6. Escritura de Stream (Kafka Sink con Micro-lotes)
    # ---------------------------------------------------------
    checkpoint_dir = os.path.join(project_root, 'pipeline', 'checkpoints_kafka')
    
    query = output_df.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "alertas-eyeccion") \
        .option("checkpointLocation", checkpoint_dir) \
        .trigger(processingTime='2 seconds') \
        .start()
        
    print("Stream Escuchando tópico 'raw-video-stream' y publicando en 'alertas-eyeccion'.")
    print("Esperando flujos de datos...")
    
    # Mantiene el proceso vivo escuchando los mensajes entrantes
    query.awaitTermination()

if __name__ == "__main__":
    main()
