def preprocess_image(image):
    """
    Preprocesar una imagen para el modelo de PyTorch:
    - Convertir a tensor.
    - Normalizar.
    - Añadir la dimensión del batch.
    """
    img_tensor = F.to_tensor(image)  # Convertir a tensor (C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # Añadir dimensión del batch
    return img_tensor
