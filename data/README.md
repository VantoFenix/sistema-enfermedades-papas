# Datos

Los datos **no se incluyen en el repositorio**. Este documento explica cómo obtenerlos y cómo generar la partición usada en todos los experimentos.

## Origen

| Campo | Valor |
|---|---|
| Nombre | Potato Dataset (tubérculos de papa sanos y con enfermedades) |
| Fuente | Mendeley Data — `https://data.mendeley.com/datasets/7vm7xskfg4/2` |
| Autores | `Samiul Islam, Tanzila Afrin` |
| Licencia | `CC BY 4.0` |
| Modalidad | Imagen RGB |
| Resolución | Variable (18 resoluciones distintas; mayoría 512×512). Se unifica a 224×224 en el preprocesamiento |
| Clases | 4 |

## Composición

| Clase | Imágenes | Proporción |
|---|---|---|
| Blackspot Bruising Care | 770 | 22.0% |
| Dry Rot Care | 1355 | 38.7% |
| Healthy Potatoes | 815 | 23.3% |
| Soft Rot Care | 560 | 16.0% |
| **Total** | **3500** | 100% |

El dataset está **desbalanceado**: Dry Rot tiene 2.4 veces más imágenes que Soft Rot.
La partición no modifica esta distribución (es estratificada y no se eliminan ni duplican imágenes); el desbalance se corrige durante el entrenamiento, solo sobre `train` (pesos por clase o sobremuestreo en el DataLoader).
El EDA (`notebooks/01_eda.ipynb`) no encontró imágenes corruptas ni duplicados exactos (hash MD5).

## Cómo obtenerlo

**Opción 1 — Carpeta compartida del curso** (pedir acceso):

```
MyDrive/PERCEPCION COMPUTACIONAL - LAB: 6013/GRUPO 1/Dataset/Potato_Dataset
```

Enlace: `https://drive.google.com/drive/folders/1IdekPFGgZwjOolva8yyALrttuhYC2H6R?usp=drive_link`

**Opción 2 — Descarga desde la fuente original:** `https://data.mendeley.com/datasets/7vm7xskfg4/2`

En ambos casos, la estructura esperada es una subcarpeta por clase:

```
Potato_Dataset/
├── Blackspot Bruising Care/
├── Dry Rot Care/
├── Healthy Potatoes/
└── Soft Rot Care/
```

## Partición

Partición estratificada por clase **70 / 15 / 15** con semilla fija `42`:

```bash
python src/datos/particion.py --origen <ruta>/Potato_Dataset --destino <ruta>/Potato_Particionado --seed 42
```

Genera:

```
Potato_Particionado/
├── train/<clase>/   # 2450 imágenes
├── val/<clase>/     # 525 imágenes
└── test/<clase>/    # 525 imágenes
```

| Clase | Train | Val | Test |
|---|---|---|---|
| Blackspot Bruising Care | 539 | 115 | 116 |
| Dry Rot Care | 948 | 204 | 203 |
| Healthy Potatoes | 571 | 122 | 122 |
| Soft Rot Care | 392 | 84 | 84 |

- Si la carpeta destino ya existe, no se vuelve a generar, para que la partición no cambie entre ejecuciones.
- Los análisis de diseño (`notebooks/02_preprocesamiento_senales.ipynb`) usan **solo `train`**.
- El conjunto `test` se reserva para la evaluación final y no interviene en decisiones de diseño.
