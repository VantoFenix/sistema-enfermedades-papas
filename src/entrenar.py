import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# Asegurar de que la raíz del proyecto esté en el path para poder importar módulos
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
    torch.cuda.manual_seed_all(seed) # Si usas multi-GPU
    
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

def main():
    # 1. Fijar semilla de reproducibilidad
    set_seed(42)
    
    # 2. Configurar dispositivo y rutas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando entrenamiento en: {device}")
    
    data_dir_train = os.path.join(project_root, 'data', 'Potato_Particionado', 'train')
    data_dir_val = os.path.join(project_root, 'data', 'Potato_Particionado', 'val')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 3. Cargar Datasets (Aplicando el preprocesamiento exitoso de CLAHE)
    print("Cargando datasets particionados y aplicando CLAHE...")
    train_dataset = PotatoDataset(data_dir=data_dir_train, use_clahe=True)
    val_dataset = PotatoDataset(data_dir=data_dir_val, use_clahe=True)
    
    # 4. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 5. Calcular pesos para manejar el desbalance
    class_weights = calculate_class_weights(train_dataset, num_classes=4).to(device)
    print(f"Pesos calculados por clase: {class_weights.cpu().numpy()}")
    
    # 6. Instanciar Modelo, Función de Pérdida y Optimizador
    model = DefectosPapaResNet(num_classes=4).to(device)
    
    # Le pasamos los pesos al CrossEntropyLoss para contrarrestar el desbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # AdamW con lr=1e-3 y weight_decay=1e-4 sobre los parámetros entrenables
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    # 7. Bucle de Entrenamiento
    max_epochs = 50
    patience = 7
    epochs_no_improve = 0
    best_f1 = 0.0
    best_model_path = os.path.join(models_dir, 'defectos_v1.pt')
    
    print("\n" + "="*50)
    print("🚀 ARRANCANDO ENTRENAMIENTO PRINCIPAL")
    print("="*50)
    
    for epoch in range(max_epochs):
        # --- Fase de Entrenamiento (Train) ---
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Época {epoch+1:02d}/{max_epochs} [Train]")
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
        
        # --- Fase de Validación (Eval) ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        val_bar = tqdm(val_loader, desc=f"Época {epoch+1:02d}/{max_epochs} [Eval ]")
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
        
        # --- Lógica de Early Stopping y Guardado ---
        if val_f1 > best_f1:
            print(f"  ⭐ ¡Nuevo mejor F1-Macro! ({best_f1:.4f} --> {val_f1:.4f}). Guardando modelo...")
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  ⚠️ Sin mejora en F1-Macro. Paciencia de Early Stopping: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                print(f"\n⏹️ Early Stopping activado. El modelo no ha mejorado durante {patience} épocas consecutivas.")
                break
                
    print("\n" + "="*50)
    print("🏁 ENTRENAMIENTO FINALIZADO")
    print("="*50)
    print(f"El mejor modelo obtuvo un F1-Macro de: {best_f1:.4f}")
    print(f"Pesos guardados exitosamente en: {best_model_path}")

if __name__ == "__main__":
    main()
