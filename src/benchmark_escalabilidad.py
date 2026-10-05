import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
from pyspark.sql.types import StringType

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet

_modelo = None

def get_model():
    global _modelo
    if _modelo is None:
        device = torch.device("cpu") # Forzar CPU para medir paralelización sin colisiones de CUDA
        model_path = os.path.join(project_root, 'models', 'defectos_v1.pt')
        _modelo = DefectosPapaResNet(num_classes=4)
        _modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _modelo = _modelo.to(device)
        _modelo.eval()
    return _modelo

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def predict_defect_udf(ids: pd.Series) -> pd.Series:
    modelo = get_model()
    predictions = []
    # Generamos un tensor dummy único para aislar puramente el tiempo del motor de inferencia
    # y evitar sobrecarga de RAM (OOM) al pasar 5000 arrays a través de PyArrow.
    dummy_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    with torch.no_grad():
        for _ in ids:
            output = modelo(dummy_tensor)
            _, pred = torch.max(output, 1)
            predictions.append(str(pred.item()))
    return pd.Series(predictions)

def main():
    N_IMAGES = 5000
    
    print("Iniciando benchmark (5000 imágenes)...")
    # 1. Python Puro (Secuencial)
    modelo = get_model()
    dummy_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32)
    start = time.time()
    with torch.no_grad():
        for _ in range(N_IMAGES):
            _ = modelo(dummy_tensor)
    time_seq = time.time() - start
    print(f"Secuencial (Python Puro): {time_seq:.2f} s")
    
    # DataFrame dummy con IDs
    df_pandas = pd.DataFrame({'id': range(N_IMAGES)})
    
    # 2. Spark Local[1]
    spark = SparkSession.builder.master("local[1]").appName("BM1").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    df_spark = spark.createDataFrame(df_pandas).repartition(1)
    start = time.time()
    df_spark.withColumn("pred", predict_defect_udf(col("id"))).count()
    time_sp1 = time.time() - start
    spark.stop()
    print(f"Spark (local[1]): {time_sp1:.2f} s")
    
    # 3. Spark Local[2]
    spark = SparkSession.builder.master("local[2]").appName("BM2").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    df_spark = spark.createDataFrame(df_pandas).repartition(2)
    start = time.time()
    df_spark.withColumn("pred", predict_defect_udf(col("id"))).count()
    time_sp2 = time.time() - start
    spark.stop()
    print(f"Spark (local[2]): {time_sp2:.2f} s")
    
    # 4. Spark Local[4]
    spark = SparkSession.builder.master("local[4]").appName("BM4").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    df_spark = spark.createDataFrame(df_pandas).repartition(4)
    start = time.time()
    df_spark.withColumn("pred", predict_defect_udf(col("id"))).count()
    time_sp4 = time.time() - start
    spark.stop()
    print(f"Spark (local[4]): {time_sp4:.2f} s")

if __name__ == '__main__':
    main()
