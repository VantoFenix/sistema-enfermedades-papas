import os
import sys
import torch
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet
from src.datos.dataset import PotatoDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir_test = os.path.join(project_root, 'data', 'Potato_Particionado', 'test')
    model_path = os.path.join(project_root, 'models', 'defectos_v1.pt')
    
    test_dataset = PotatoDataset(data_dir=data_dir_test, use_clahe=False)
    
    model = DefectosPapaResNet(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    results = {
        'Baja': {'true': [], 'pred': []},
        'Media': {'true': [], 'pred': []},
        'Alta': {'true': [], 'pred': []}
    }
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img_path = test_dataset.image_paths[i]
            
            # Cargar imagen cruda con OpenCV para calcular brillo
            img_raw = cv2.imread(img_path)
            if img_raw is None:
                # Si falla cv2 por el bug de PIL, usamos PIL
                from PIL import Image
                img_pil = Image.open(img_path).convert('L')
                avg_intensity = np.mean(np.array(img_pil))
            else:
                gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
                avg_intensity = np.mean(gray)
            
            if avg_intensity < 85:
                cat = 'Baja'
            elif avg_intensity <= 170:
                cat = 'Media'
            else:
                cat = 'Alta'
                
            tensor, label = test_dataset[i]
            input_tensor = tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            
            results[cat]['true'].append(label)
            results[cat]['pred'].append(pred.item())
            
    print("--- Resultados de Evaluación por Iluminación ---")
    for cat in ['Baja', 'Media', 'Alta']:
        y_true = results[cat]['true']
        y_pred = results[cat]['pred']
        n = len(y_true)
        if n > 0:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            print(f"Categoría {cat} (N={n}): Accuracy = {acc:.4f} | F1-Macro = {f1:.4f}")
        else:
            print(f"Categoría {cat} (N={n}): Sin muestras")

if __name__ == '__main__':
    main()
