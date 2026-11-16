import io
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Asegurar importación de src desde la raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modelos.arquitectura import DefectosPapaResNet

# Clases en orden alfabético de ImageFolder
CLASES = ['Blackspot Bruising', 'Dry Rot', 'Healthy Potatoes', 'Soft Rot']

# Estado global de la aplicación (Singleton en memoria)
app_state: Dict[str, Any] = {}

def get_transforms():
    """Transformaciones estándar de preprocesamiento e inferencia de ImageNet."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: carga el modelo una sola vez al iniciar."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.getenv("MODEL_PATH", os.path.join(project_root, "models", "defectos_v1.pt"))
    
    # Fallback si se ejecuta en contenedor Docker
    if not os.path.exists(model_path) and os.path.exists("/app/models/defectos_v1.pt"):
        model_path = "/app/models/defectos_v1.pt"
        
    print(f"[INFO] Inicializando servicio. Dispositivo: {device} | Ruta del modelo: {model_path}")
    
    modelo = DefectosPapaResNet(num_classes=len(CLASES))
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        modelo.load_state_dict(checkpoint)
        modelo = modelo.to(device)
        modelo.eval()
        app_state["model"] = modelo
        app_state["model_loaded"] = True
        print("[INFO] Modelo cargado exitosamente en memoria en modo eval.")
    else:
        print(f"[WARN] No se encontró el archivo de pesos en {model_path}.")
        app_state["model"] = None
        app_state["model_loaded"] = False
        
    app_state["device"] = device
    app_state["transform"] = get_transforms()
    
    yield
    
    app_state.clear()
    print("[INFO] Servicio detenido y recursos liberados.")

app = FastAPI(
    title="HotPotato - API de Diagnóstico de Tubérculos de Papa",
    description="API REST para detección e inferencia de defectos en papas usando ResNet18 (PyTorch)",
    version="1.0.0",
    lifespan=lifespan
)

# Configuración de CORS para integración con clientes web y dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Endpoint de monitoreo y health check."""
    return {
        "status": "ok",
        "model": "HotPotato-ResNet18",
        "version": "1"
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen (JPEG/PNG), aplica transformaciones estándar de ImageNet
    y devuelve la predicción con el desglose de probabilidades por clase.
    """
    if not app_state.get("model_loaded", False) or app_state.get("model") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo predictivo no se encuentra cargado en memoria."
        )
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al decodificar la imagen: {str(e)}"
        )
        
    transform = app_state["transform"]
    device = app_state["device"]
    modelo = app_state["model"]
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = modelo(input_tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=0)
        
    clase_predicha = CLASES[pred_idx.item()]
    confianza = float(conf.item())
    probabilidades = {clase: float(probs[i].item()) for i, clase in enumerate(CLASES)}
    
    return {
        "clase_predicha": clase_predicha,
        "confianza": round(confianza, 4),
        "probabilidades": {k: round(v, 4) for k, v in probabilidades.items()},
        "modelo_version": "HotPotato-ResNet18:v1"
    }
