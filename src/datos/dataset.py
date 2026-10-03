import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PotatoDataset(Dataset):
    """
    Dataset PyTorch personalizado para clasificación de papas.
    Permite evaluar el impacto del preprocesamiento CLAHE.
    """
    def __init__(self, data_dir, use_clahe=False):
        self.data_dir = data_dir
        self.use_clahe = use_clahe
        self.image_paths = []
        self.labels = []
        
        # Obtener y ordenar las clases del directorio
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
        # Transformaciones: redimensionar, a tensor y normalizar con ImageNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Leer imagen con PIL para mejor compatibilidad (algunos archivos pueden ser GIF disfrazados)
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}. Error: {e}")
            
        if not self.use_clahe:
            image = pil_image
        else:
            # Convertir a numpy array para OpenCV
            image_np = np.array(pil_image)
            
            # Convertir RGB al espacio Lab
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Aplicar ecualización adaptativa (CLAHE) únicamente al canal L
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Unir los canales y convertir de nuevo a RGB
            limg = cv2.merge((cl, a, b))
            image_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Volver a formato PIL
            image = Image.fromarray(image_np)
            
        # Aplicar redimensionamiento y normalización
        image = self.transform(image)
            
        return image, label

if __name__ == '__main__':
    import sys
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score, f1_score
    
    # Añadir raíz del proyecto a sys.path para poder importar DefectosPapaResNet
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    try:
        from src.modelos.arquitectura import DefectosPapaResNet
    except ImportError as e:
        print("Error al importar la arquitectura. Asegúrate de ejecutar desde la raíz del proyecto.")
        raise e

    def train_and_eval(use_clahe, data_dir_train, data_dir_val, epochs=5):
        print(f"\n--- Iniciando entrenamiento con use_clahe={use_clahe} ---")
        train_dataset = PotatoDataset(data_dir=data_dir_train, use_clahe=use_clahe)
        val_dataset = PotatoDataset(data_dir=data_dir_val, use_clahe=use_clahe)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo a utilizar: {device}")
        
        # Instanciar el modelo ResNet
        model = DefectosPapaResNet(num_classes=4).to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Optimizamos solo los parámetros que requieren gradiente (bloque 4 y la FC)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Evaluación
            model.eval()
            all_preds = []
            all_labels = []
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            
            print(f"Época {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val F1-Macro: {f1:.4f}")
            
        print(f"Resultados Finales (use_clahe={use_clahe}): Exactitud = {acc:.4f}, F1-Macro = {f1:.4f}")
        return acc, f1

    # Rutas al dataset particionado
    base_data_dir_train = os.path.join(project_root, 'data', 'Potato_Particionado', 'train')
    base_data_dir_val = os.path.join(project_root, 'data', 'Potato_Particionado', 'val')
    
    if not os.path.exists(base_data_dir_train):
        print(f"Error: No se encontró el directorio de entrenamiento en {base_data_dir_train}")
        sys.exit(1)
        
    print(f"Usando directorio de entrenamiento: {base_data_dir_train}")
    
    # 1. Entrenar y evaluar con el dataset original
    acc_raw, f1_raw = train_and_eval(use_clahe=False, data_dir_train=base_data_dir_train, data_dir_val=base_data_dir_val, epochs=5)
    
    # 2. Entrenar y evaluar aplicando CLAHE
    acc_clahe, f1_clahe = train_and_eval(use_clahe=True, data_dir_train=base_data_dir_train, data_dir_val=base_data_dir_val, epochs=5)
    
    print("\n" + "="*50)
    print("RESUMEN DE COMPARACIÓN")
    print("="*50)
    print(f"Dataset Crudo (RGB) - Exactitud: {acc_raw:.4f} | F1-Macro: {f1_raw:.4f}")
    print(f"Dataset con CLAHE   - Exactitud: {acc_clahe:.4f} | F1-Macro: {f1_clahe:.4f}")
    print("="*50)
