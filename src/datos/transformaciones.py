from torchvision import transforms

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

# Solo entrenamiento: el tubérculo puede llegar en cualquier orientación y con
# iluminación variable; rotar o voltear(horizontal o vertical) no cambia el 
# tipo de enfermedad.
aumento = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomVerticalFlip(p=0.5),
])

transform_train = transforms.Compose([
    aumento,
    transforms.ToTensor(),
    transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
])

# Validación y prueba: sin aleatoriedad, misma normalización
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEDIA_IMAGENET, STD_IMAGENET),
])