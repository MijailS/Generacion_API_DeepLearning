import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def load_model():
    """
    Carga el modelo Faster R-CNN entrenado desde un archivo .pth.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Utilizar los pesos explícitos de torchvision
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)  # Especificar los pesos aquí

    # Ajustar la salida del modelo para las clases personalizadas
    num_classes = 5  # Fondo + manzanas verdes buenas + manzanas rojas buenas + manzanas verdes malas + manzanas rojas malas
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(
    torch.load("models/fasterrcnn_coco_apples_gpu.pth",map_location=device))
    model.to(device)
    model.eval()  # Configurar en modo evaluación

    return model, device