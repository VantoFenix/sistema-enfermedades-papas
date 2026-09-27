import os
import shutil
from sklearn.model_selection import train_test_split


def crear_particion(origen, destino, seed=42):
    ##Partición estratificada 70/15/15 copiada en destino/{train,val,test}/<clase>.
    if os.path.exists(destino):
        return

    rutas, etiquetas = [], []
    for clase in sorted(os.listdir(origen)):
        ruta_clase = os.path.join(origen, clase)
        for archivo in sorted(os.listdir(ruta_clase)):
            rutas.append(os.path.join(ruta_clase, archivo))
            etiquetas.append(clase)

    X_train, X_temp, y_train, y_temp = train_test_split(
        rutas, etiquetas, test_size=0.30, stratify=etiquetas, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    for split, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        for ruta, clase in zip(X, y):
            carpeta = os.path.join(destino, split, clase)
            os.makedirs(carpeta, exist_ok=True)
            shutil.copy2(ruta, carpeta)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Partición estratificada 70/15/15')
    parser.add_argument('--origen', required=True, help='Carpeta del dataset original (una subcarpeta por clase)')
    parser.add_argument('--destino', required=True, help='Carpeta donde se crean train/val/test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    crear_particion(args.origen, args.destino, args.seed)