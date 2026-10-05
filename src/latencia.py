import os
import sys
import time
import cv2
import glob
import numpy as np
import torch
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet
from src.datos.dataset import PotatoDataset

def extract_features(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_resized = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys',
                       visualize=False, feature_vector=True)
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_img], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_img], [2], None, [16], [0, 256]).flatten()
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    return np.concatenate([hog_features, color_features])

def main():
    data_dir_train = os.path.join(project_root, 'data', 'Potato_Particionado', 'train')
    
    # Tomar 1 imagen de prueba
    classes = os.listdir(data_dir_train)
    sample_img_path = glob.glob(os.path.join(data_dir_train, classes[0], '*.*'))[0]
    
    # ----------------------------------------------------
    # 1. Medir ResNet18
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_resnet = DefectosPapaResNet(num_classes=4)
    model_path = os.path.join(project_root, 'models', 'defectos_v1.pt')
    model_resnet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model_resnet = model_resnet.to(device)
    model_resnet.eval()
    
    # Cargar usando la transformación del dataset
    dataset = PotatoDataset(data_dir=data_dir_train, use_clahe=False)
    # Obtenemos la tupla (imagen_tensor, label) de la primera muestra
    input_tensor, _ = dataset[0]
    input_tensor = input_tensor.unsqueeze(0).to(device) # Añadir dimensión de batch
    
    # Warm-up ResNet
    with torch.no_grad():
        for _ in range(5):
            _ = model_resnet(input_tensor)
            
    # Medir ResNet (100 iteraciones para promediar)
    n_iters = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model_resnet(input_tensor)
    resnet_time_ms = ((time.time() - start_time) / n_iters) * 1000
    
    # ----------------------------------------------------
    # 2. Medir SVM + HOG/HSV
    # ----------------------------------------------------
    # Entrenar un SVM rápido con unas pocas imágenes para poder hacer predict real
    X_train, y_train = [], []
    for c_idx, c_name in enumerate(classes):
        imgs = glob.glob(os.path.join(data_dir_train, c_name, '*.*'))[:10]
        for img in imgs:
            f = extract_features(img)
            if f is not None:
                X_train.append(f)
                y_train.append(c_idx)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=10, gamma='scale'))
    ])
    pipeline.fit(X_train, y_train)
    
    # Warm-up SVM pipeline (extracción + predict)
    for _ in range(5):
        feat = extract_features(sample_img_path)
        _ = pipeline.predict([feat])
        
    # Medir SVM
    start_time = time.time()
    for _ in range(n_iters):
        feat = extract_features(sample_img_path)
        _ = pipeline.predict([feat])
    svm_time_ms = ((time.time() - start_time) / n_iters) * 1000
    
    # ----------------------------------------------------
    # Salida estricta
    # ----------------------------------------------------
    print(f"ResNet18: {resnet_time_ms:.2f} ms")
    print(f"Modelo Clasico (SVM + HOG/HSV): {svm_time_ms:.2f} ms")

if __name__ == '__main__':
    main()
