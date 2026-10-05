import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet
from src.datos.dataset import PotatoDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir_test = os.path.join(project_root, 'data', 'Potato_Particionado', 'test')
    model_path = os.path.join(project_root, 'models', 'defectos_v1.pt')
    
    # Cargar dataset de test (Ruta RGB cruda)
    test_dataset = PotatoDataset(data_dir=data_dir_test, use_clahe=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    num_samples = len(test_dataset)
    print(f"Muestras en conjunto de TEST: {num_samples}")
    
    # Instanciar y cargar pesos
    model = DefectosPapaResNet(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Classification Report
    target_names = test_dataset.classes
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # Matriz de Confusión Normalizada
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".4f", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Matriz de Confusión Normalizada - TEST")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    
    fig_dir = os.path.join(project_root, 'figuras')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'matriz_confusion_test.png')
    plt.savefig(fig_path)
    print(f"\nMatriz de confusión guardada en: {fig_path}")

if __name__ == '__main__':
    main()
