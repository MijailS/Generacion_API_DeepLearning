
def process_detections(outputs, confidence_threshold=0.65):
    """
    Procesar las salidas del modelo y filtrar detecciones por confianza.
    """
    boxes = outputs["boxes"]
    scores = outputs["scores"]
    labels = outputs["labels"]

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            detections.append({
                "box": box.tolist(),
                "score": score.item(),
                "label": label.item()
            })

    return detections

def count_apples(detections, confidence_threshold=0.8):
    """
    Contar manzanas rojas y verdes basadas en las etiquetas detectadas.
    """
    red_apples = 0
    green_apples = 0
    bad_green_apples = 0
    bad_red_apples = 0

    for detection in detections:
        label = detection["label"]
        score = detection["score"]

        # Filtrar detecciones con baja confianza
        if score < confidence_threshold:
            continue

        # Etiqueta 1: Manzanas rojas
        if label == 1:
            red_apples += 1

        # Etiqueta 2: Manzanas verdes
        elif label == 2:
            green_apples += 1# Etiqueta 2: Manzanas verdes

        elif label == 3:
            bad_green_apples += 1# Etiqueta 2: Manzanas verdes

        elif label == 2:
            bad_red_apples += 1

    return red_apples, green_apples
