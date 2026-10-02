import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary

class DefectosPapaResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DefectosPapaResNet, self).__init__()
        
        # 1. Cargar resnet18 preentrenada con los pesos más recientes
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 2. Transfer learning: Congelar los bloques 1 al 3 (y capas iniciales)
        # Primero congelamos todo el modelo
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Luego dejamos entrenable (descongelamos) únicamente el bloque 4
        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
        # 3. Modificar la capa final (fc) por el Sequential solicitado
        num_ftrs = self.model.fc.in_features # En ResNet18 esto es 512
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Instanciar el modelo
    modelo = DefectosPapaResNet(num_classes=4)
    
    # Imprimir estructura con torchinfo simulando una entrada (Batch=1, Canales=3, H=224, W=224)
    print("Resumen de la arquitectura DefectosPapaResNet:")
    # Usamos col_names para ver claramente qué capas son entrenables
    summary(modelo, input_size=(1, 3, 224, 224), 
            col_names=["input_size", "output_size", "num_params", "trainable"])
