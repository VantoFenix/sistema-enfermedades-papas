import os
import sys
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# Asegurar que la raíz del proyecto esté en el path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet
from src.datos.dataset import PotatoDataset

def set_seed(seed=42):
    """
    Fija todas las semillas posibles para garantizar la reproducibilidad
    de las ejecuciones (mismo shuffle, misma inicialización de pesos, etc).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configuraciones para hacer CuDNN determinista
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_class_weights(dataset, num_classes=4):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de clases.
    Se asume que el 'dataset' pasado ya es exclusivamente el de entrenamiento.
    """
    train_labels = dataset.labels
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)
    
    # Fórmula: Peso = N_total / (N_clases * N_muestras_clase)
    weights = total_samples / (num_classes * class_counts)
    return torch.FloatTensor(weights)

def ejecutar_corrida(config: dict, device: torch.device, models_dir: str):
    """
    Ejecuta un ciclo completo de entrenamiento y validación para una configuración dada,
    registrando hiperparámetros, métricas por época y el modelo en MLflow.
    Guarda los mejores pesos en un archivo temporal propio: models/temp_{run_name}.pt
    """
    run_name = config["run_name"]
    print("\n" + "="*60)
    print(f"🚀 INICIANDO CORRIDA: {run_name}")
    print(f"Hiperparámetros: {config}")
    print("="*60)
    
    # 1. Fijar semilla al inicio de cada corrida para aislamiento determinista
    set_seed(42)
    
    data_dir_train = os.path.join(project_root, 'data', 'Potato_Particionado', 'train')
    data_dir_val = os.path.join(project_root, 'data', 'Potato_Particionado', 'val')
    
    # 2. Cargar Datasets con o sin CLAHE según configuración
    train_dataset = PotatoDataset(data_dir=data_dir_train, use_clahe=config["use_clahe"])
    val_dataset = PotatoDataset(data_dir=data_dir_val, use_clahe=config["use_clahe"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    
    # 3. Pesos para clases desbalanceadas
    class_weights = calculate_class_weights(train_dataset, num_classes=4).to(device)
    
    # 4. Instanciar Modelo, Función de Pérdida y Optimizador
    model = DefectosPapaResNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    if config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr"],
            momentum=0.9,
            weight_decay=config["weight_decay"]
        )
    else:
        raise ValueError(f"Optimizador no soportado: {config['optimizer']}")
        
    temp_model_path = os.path.join(models_dir, f"temp_{run_name}.pt")
    
    best_f1 = 0.0
    epochs_no_improve = 0
    patience = config["patience"]
    max_epochs = config.get("max_epochs", 50)
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        # Registrar hiperparámetros
        mlflow.log_params({
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "optimizer": config["optimizer"],
            "batch_size": config["batch_size"],
            "use_clahe": config["use_clahe"],
            "patience": config["patience"],
            "max_epochs": max_epochs,
            "seed": 42,
            "architecture": "DefectosPapaResNet"
        })
        
        for epoch in range(max_epochs):
            # --- Fase de Entrenamiento ---
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_loader, desc=f"[{run_name}] Época {epoch+1:02d}/{max_epochs} [Train]")
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_bar.set_postfix(loss=loss.item())
                
            train_loss /= len(train_loader.dataset)
            
            # --- Fase de Validación ---
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            val_bar = tqdm(val_loader, desc=f"[{run_name}] Época {epoch+1:02d}/{max_epochs} [Eval ]")
            with torch.no_grad():
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            
            print(f"Resumen Época {epoch+1:02d}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1-Macro: {val_f1:.4f}")
            
            # Registrar métricas en MLflow por época
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_macro": val_f1
            }, step=epoch + 1)
            
            # Guardado en archivo temporal exclusivo de la corrida
            if val_f1 > best_f1:
                print(f"  ⭐ ¡Nuevo mejor F1-Macro! ({best_f1:.4f} --> {val_f1:.4f}). Guardando pesos en {temp_model_path}...")
                best_f1 = val_f1
                torch.save(model.state_dict(), temp_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  ⚠️ Sin mejora en F1-Macro. Paciencia de Early Stopping: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print(f"\n⏹️ Early Stopping activado en época {epoch+1}.")
                    break
                    
        # Cargar los mejores pesos encontrados en esta corrida para registrar el artefacto final
        if os.path.exists(temp_model_path):
            model.load_state_dict(torch.load(temp_model_path, map_location=device, weights_only=True))
            model.eval()
            
        # Mover el modelo a CPU para evitar colisiones de dispositivos en FakeTensor de PyTorch/MLflow
        model_cpu = model.to("cpu")
        model_cpu.eval()
        
        # Generar firma e inferencia de prueba en CPU
        dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
        with torch.no_grad():
            dummy_output = model_cpu(dummy_input).cpu().numpy()
        ejemplo_entrada = dummy_input.cpu().numpy()
        signature = infer_signature(ejemplo_entrada, dummy_output)
        
        # Registrar el modelo de esta corrida como artefacto en MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            name="model",
            signature=signature,
            input_example=ejemplo_entrada
        )
        mlflow.log_metric("best_val_f1_macro", best_f1)
        
        print(f"🏁 Corrida {run_name} finalizada. Mejor F1: {best_f1:.4f}")
        
        return {
            "run_name": run_name,
            "run_id": run_id,
            "best_val_f1": best_f1,
            "temp_model_path": temp_model_path
        }

def main():
    # 1. Configurar Experimento de MLflow
    experiment_name = "HotPotato_Defect_Classification"
    mlflow.set_experiment(experiment_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware detectado para experimentación: {device}")
    
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Definición de las 4 corridas comparativas requeridas
    corridas = [
        {
            "run_name": "Corrida_1_Ganadora",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "batch_size": 32,
            "use_clahe": False,
            "patience": 7,
            "max_epochs": 50
        },
        {
            "run_name": "Corrida_2_Ablacion_CLAHE",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "batch_size": 32,
            "use_clahe": True,
            "patience": 7,
            "max_epochs": 50
        },
        {
            "run_name": "Corrida_3_LR_Bajo",
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "batch_size": 32,
            "use_clahe": False,
            "patience": 7,
            "max_epochs": 50
        },
        {
            "run_name": "Corrida_4_Optimizador_Clasico",
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "optimizer": "SGD",
            "batch_size": 32,
            "use_clahe": False,
            "patience": 7,
            "max_epochs": 50
        }
    ]
    
    resultados = []
    
    # 3. Ejecutar las 4 corridas comparativas
    for config in corridas:
        res = ejecutar_corrida(config, device, models_dir)
        resultados.append(res)
        
    # 4. Evaluación comparativa: encontrar la corrida con mayor F1-Macro
    mejor_corrida = max(resultados, key=lambda x: x["best_val_f1"])
    
    print("\n" + "="*60)
    print("🏆 RESUMEN FINAL DE COMPARACIÓN DE HIPERPARÁMETROS")
    print("="*60)
    for res in resultados:
        print(f"  - {res['run_name']}: Mejor Val F1-Macro = {res['best_val_f1']:.4f} (ID: {res['run_id']})")
    print("-" * 60)
    print(f"🥇 Corrida Ganadora: {mejor_corrida['run_name']} con F1 = {mejor_corrida['best_val_f1']:.4f}")
    
    # 5. Copiar los pesos de la mejor corrida a models/defectos_v1.pt
    destino_final = os.path.join(models_dir, 'defectos_v1.pt')
    shutil.copy2(mejor_corrida["temp_model_path"], destino_final)
    print(f"💾 Pesos óptimos guardados exitosamente en: {destino_final}")
    
    # 6. Registrar en el Model Registry bajo "HotPotato-ResNet18"
    model_uri = f"runs:/{mejor_corrida['run_id']}/model"
    print(f"📦 Registrando modelo ganador en Model Registry ('HotPotato-ResNet18') desde {model_uri}...")
    try:
        reg_model = mlflow.register_model(model_uri=model_uri, name="HotPotato-ResNet18")
        print(f"✅ Modelo registrado satisfactoriamente: {reg_model.name} (Versión {reg_model.version})")
    except Exception as e:
        print(f"⚠️ Nota de registro (verifique si el backend store de MLflow soporta Model Registry): {e}")
        
    print("\nProceso de experimentación finalizado. Inicie la UI con: mlflow ui --port 5000")

if __name__ == "__main__":
    main()
