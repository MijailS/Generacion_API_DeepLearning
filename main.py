from flask import Flask, request, jsonify
from PIL import Image
import torch
import cv2
import time
from torchvision.transforms import functional as F
from models import faster_rcnn
import numpy as np


# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo al iniciar el servidor
model, device = faster_rcnn.load_model()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint para procesar una imagen y realizar la predicción.
    """
    try:
        # Validar si se recibe un archivo
        if "image" not in request.files:
            return jsonify({"error": "Se requiere una imagen para realizar la predicción"}), 400

        # Leer el archivo enviado
        file = request.files["image"]

        # Validar el formato del archivo
        if not file or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Formato de archivo no soportado. Use .png, .jpg o .jpeg"}), 400

        # Leer la imagen en OpenCV
        file.stream.seek(0)  # Reiniciar el puntero del archivo
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "No se pudo leer la imagen. Verifique el formato y el contenido"}), 400

        # Convertir la imagen a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocesar la imagen
        tensor = F.to_tensor(Image.fromarray(frame)).unsqueeze(0).to(device)

        # Realizar predicciones
        start_time = time.time()
        with torch.no_grad():
            outputs = model(tensor)

        # Inicializar contadores
        cont_verde_buena = 0
        cont_roja_buena = 0
        cont_roja_mala = 0
        cont_verde_mala = 0

        # Dibujar cajas y procesar detecciones
        detections = []
        tipo_manzana= 0
        for box, score, label in zip(outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]):
            if score > 0.75:  # Filtrar predicciones con confianza > 0.75
                x1, y1, x2, y2 = map(int, box.tolist())
                color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Verde para manzanas verdes, rojo para rojas
                if label == 1:
                    tipo_manzana ="Verde Buena"
                if label == 2:
                    tipo_manzana ="Roja Buena"
                if label == 3:
                    tipo_manzana ="Verde Mala"
                if label == 4:
                    tipo_manzana ="Roja Mala"

                # Guardar la detección
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "label": tipo_manzana,
                    "score": round(score.item(), 2)
                })

                # Incrementar contadores
                if label == 1:
                    cont_verde_buena += 1
                elif label == 2:
                    cont_roja_buena += 1
                elif label == 3:
                    cont_verde_mala += 1
                elif label == 4:
                    cont_roja_mala += 1

        # Calcular tiempo transcurrido
        elapsed_time = time.time() - start_time

        # Construir la respuesta
        response = {
            "detecciones": detections,
            "manzanas_verdes_buenas": cont_verde_buena,
            "manzanas_rojas_buenas": cont_roja_buena,
            "manzanas_rojas_malas": cont_roja_mala,
            "manzanas_verdes_malas": cont_verde_mala,
            "total_apples": cont_roja_mala+cont_verde_mala+ cont_verde_buena+cont_roja_buena,
            "processing_time": round(elapsed_time, 2)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Error durante la predicción", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
