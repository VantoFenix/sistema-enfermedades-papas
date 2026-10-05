import os
import sys
import shutil
import torch

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
    classes = test_dataset.classes
    
    # Índices de las clases
    idx_soft_rot = classes.index('Soft Rot Care')
    idx_dry_rot = classes.index('Dry Rot Care')
    idx_blackspot = classes.index('Blackspot Bruising Care')
    idx_healthy = classes.index('Healthy Potatoes')
    
    model = DefectosPapaResNet(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    fig_dir = os.path.join(project_root, 'figuras')
    os.makedirs(fig_dir, exist_ok=True)
    
    found_1, found_2, found_3 = False, False, False
    
    print("Buscando errores de clasificación específicos...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if found_1 and found_2 and found_3:
                break
                
            tensor, label = test_dataset[i]
            input_tensor = tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            
            real = label
            predicted = pred.item()
            
            img_path = test_dataset.image_paths[i]
            
            # Condición 1: Real 'Soft Rot' -> Predicha 'Dry Rot'
            if not found_1 and real == idx_soft_rot and predicted == idx_dry_rot:
                shutil.copy(img_path, os.path.join(fig_dir, 'error_1.png'))
                found_1 = True
                print("Error 1 guardado: Real 'Soft Rot' -> Predicha 'Dry Rot'")
                
            # Condición 2: Real 'Blackspot Bruising' -> Predicha 'Healthy Potatoes'
            elif not found_2 and real == idx_blackspot and predicted == idx_healthy:
                shutil.copy(img_path, os.path.join(fig_dir, 'error_2.png'))
                found_2 = True
                print("Error 2 guardado: Real 'Blackspot Bruising' -> Predicha 'Healthy Potatoes'")
                
            # Condición 3: Real 'Dry Rot' -> Predicha 'Soft Rot'
            elif not found_3 and real == idx_dry_rot and predicted == idx_soft_rot:
                shutil.copy(img_path, os.path.join(fig_dir, 'error_3.png'))
                found_3 = True
                print("Error 3 guardado: Real 'Dry Rot' -> Predicha 'Soft Rot'")

    if not found_1: print("No se encontró imagen para Error 1 (el modelo no cometió este error)")
    if not found_2: print("No se encontró imagen para Error 2 (el modelo no cometió este error)")
    if not found_3: print("No se encontró imagen para Error 3 (el modelo no cometió este error)")

if __name__ == '__main__':
    main()
