import cv2
import numpy as np
from skimage.feature import hog

bins_por_canal = [180, 256, 256]

def preprocesar_imagen(img_bgr):
    ##Aplicar CLAHE en canal L (lab) y redimensionar 224 x 224.
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    return img_bgr

def extraer_features(img_bgr):
    ##Extrae HOG (26244 dims) + histograma HSV con máscara Otsu (692 dims).
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vector_hog = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=False)
    _, mascara = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(mascara) > mascara.size * 0.7:
        mascara = cv2.bitwise_not(mascara)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    vector_hsv = []
    for c in range(3):
        hist = cv2.calcHist([img_hsv], [c], mascara,
                            [bins_por_canal[c]], [0, bins_por_canal[c]]).flatten()
        total = hist.sum()
        if total > 0:
            hist = hist / total
        vector_hsv.append(hist)
    vector_hsv = np.concatenate(vector_hsv)
    return np.concatenate([vector_hog, vector_hsv])
