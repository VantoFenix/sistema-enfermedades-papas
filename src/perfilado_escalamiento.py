import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Aseguramos que la raíz del proyecto esté en el path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet
from src.datos.dataset import PotatoDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando perfilado en dispositivo: {device}")
    
    if device.type != 'cuda':
        print("ADVERTENCIA: No se detectó GPU CUDA. 'tiempo_gpu' medirá el cómputo en CPU.")

    data_dir = os.path.join(project_root, 'data', 'Potato_Particionado', 'train')
    
    # 1. Instanciar dataset y DataLoader SIN optimizar (cuello de botella inducido)
    print("Cargando dataset...")
    dataset = PotatoDataset(data_dir=data_dir, use_clahe=False) # Usamos RGB normal por rapidez
    
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0,      # <-- Cero paralelismo en CPU para lectura
        pin_memory=False    # <-- Sin usar memoria fijada (transferencia lenta CPU -> GPU)
    )
    
    # 2. Instanciar modelo
    model = DefectosPapaResNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    # Usamos un optimizador rápido (solo de prueba para el perfilado)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    
    tiempo_io = 0.0
    tiempo_gpu = 0.0
    
    max_iteraciones = 20
    iteraciones_completadas = 0
    
    print(f"\nEjecutando perfilado exhaustivo durante {max_iteraciones} iteraciones (batches)...")
    
    # Convertimos el dataloader a un iterador para controlar manualmente el ciclo
    dataloader_iterator = iter(dataloader)
    
    # Inicializamos el cronómetro para medir la primera carga (fuera del GPU)
    t0 = time.time()
    
    for i in range(max_iteraciones):
        # -------------------------------------------------------------
        # 1. MEDIR TIEMPO DE E/S (I/O DEL DATALOADER)
        # -------------------------------------------------------------
        # Incluye la lectura física del disco SSD/HDD, la decodificación JPG (PIL),
        # las transformaciones de imagen y el empaquetado del batch.
        try:
            inputs, labels = next(dataloader_iterator)
        except StopIteration:
            break
            
        t1 = time.time()
        tiempo_io += (t1 - t0)
        
        # -------------------------------------------------------------
        # 2. MEDIR TIEMPO DE CÓMPUTO (TRANSFERENCIA A GPU + FORWARD/BACKWARD)
        # -------------------------------------------------------------
        # Sincronizamos cuda antes de medir si estamos en GPU para asegurar
        # que no se arrastren operaciones asíncronas anteriores.
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t_gpu_start = time.time()
        
        # Transferencia CPU -> GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Ciclo forward / backward / update
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Sincronizamos al terminar para garantizar que el cómputo finalizó
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        t_gpu_end = time.time()
        tiempo_gpu += (t_gpu_end - t_gpu_start)
        
        # Reiniciar cronómetro para la próxima carga de I/O
        iteraciones_completadas += 1
        t0 = time.time()

    # Calcular porcentajes finales
    tiempo_total = tiempo_io + tiempo_gpu
    pct_io = (tiempo_io / tiempo_total) * 100 if tiempo_total > 0 else 0
    pct_gpu = (tiempo_gpu / tiempo_total) * 100 if tiempo_total > 0 else 0

    print("\n" + "="*50)
    print("RESULTADOS DEL PERFILADO (MEDICIÓN DE CUELLO DE BOTELLA)")
    print("="*50)
    print(f"Iteraciones perfiladas: {iteraciones_completadas}")
    print(f"Tiempo Total consumido: {tiempo_total:.4f} segundos\n")
    print(f"Tiempo de E/S (DataLoader): {tiempo_io:.4f} s  --> [{pct_io:.1f}% del total]")
    print(f"Tiempo de Cómputo (GPU):    {tiempo_gpu:.4f} s  --> [{pct_gpu:.1f}% del total]")
    print("="*50)
    
    if pct_io > pct_gpu:
        print("DIAGNÓSTICO: Tu entrenamiento sufre de Inanición de GPU (GPU Starvation).")
        print("El modelo espera ocioso la mayor parte del tiempo a que lleguen los datos.")
    else:
        print("DIAGNÓSTICO: Tu pipeline está limitado por el cómputo (GPU Bound).")
        print("Considera optimizar la arquitectura, usar Mixed Precision o Data Parallelism.")

if __name__ == '__main__':
    main()


# ==============================================================================
# 🚀 PROPUESTA PARA TU INFORME: ESCALAMIENTO Y OPTIMIZACIÓN
# ==============================================================================
"""
A continuación, se presenta la solución comentada para la optimización propuesta
en el reporte de escalamiento, solucionando cuellos de botella de E/S y 
habilitando el entrenamiento distribuido en múltiples GPUs.

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distribuido(rank, world_size):
    # Inicializar el grupo de procesos para comunicación entre GPUs (backend nccl)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def entrenamiento_optimizado(rank, world_size):
    setup_distribuido(rank, world_size)
    torch.cuda.set_device(rank)

    # 1. OPTIMIZACIÓN DE E/S (DataLoader optimizado)
    # ----------------------------------------------
    dataset = PotatoDataset(data_dir=data_dir, use_clahe=False)
    
    # El Sampler distribuye el dataset equitativamente entre todas las GPUs sin duplicaciones
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader_optimizado = DataLoader(
        dataset, 
        batch_size=64,      # Batch más grande si hay VRAM disponible
        shuffle=False,      # IMPORTANTE: DistributedSampler ya se encarga de barajar
        sampler=sampler,
        num_workers=4,      # <-- RESOLUCIÓN DEL CUELLO E/S: Usa múltiples núcleos CPU para leer imágenes
        pin_memory=True,    # <-- RESOLUCIÓN DEL CUELLO E/S: Fija la memoria permitiendo DMA asíncrono súper rápido a GPU
        prefetch_factor=2   # <-- Empieza a leer los siguientes 2 batches mientras la GPU sigue ocupada
    )

    # 2. OPTIMIZACIÓN DE GPU (Paralelismo de Datos)
    # ----------------------------------------------
    model = DefectosPapaResNet(num_classes=4).to(rank)
    
    # Envolver el modelo en DistributedDataParallel (DDP)
    # Esto orquesta el Ring All-Reduce, promediando los gradientes de todas 
    # las GPUs en cada paso de retropropagación automáticamente.
    model = DDP(model, device_ids=[rank])
    
    # (El bucle de entrenamiento continuaría aquí, usando el dataloader_optimizado)
    
    dist.destroy_process_group()
"""
