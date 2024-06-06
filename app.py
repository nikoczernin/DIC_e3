from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import uuid
import os
from Yolo_Flask import Yolo

app = Flask(__name__)

# Initialize the Yolo object
yolo = Yolo()

@app.route('/object_detection', methods=['POST'])
def object_detection():
    data = request.json
    image_id = data.get('id')
    image_data = data.get('image_data')

    try:
        image_np = _decode_image(image_data)
        img_path = _save_image(image_np, image_id)
        class_ids, confidences = yolo.transform(img_path)
        detected_objects = _format_detection_results(class_ids, confidences)
        response = {
            "id": image_id,
            "objects": detected_objects
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

def _decode_image(image_data):
    image_bytes = base64.b64decode(image_data)
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decoding failed")
    return img

def _save_image(img, image_id):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, f"{image_id}.jpg")
    cv2.imwrite(img_path, img)
    return img_path

def _format_detection_results(class_ids, confidences):
    detected_objects = [
        {"label": yolo.classes[class_ids[i]], "accuracy": confidences[i]}
        for i in range(len(class_ids))
    ]
    return detected_objects

@app.route('/test', methods=['GET'])
def test_local_execution():
    input_folder = "input_folder"
    combined_results = []

    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)
        if not os.path.isfile(image_path):
            print(f"No file: {image_path}")
            continue

        encoded_string = _encode_image(image_path)
        payload = {
            "id": str(uuid.uuid4()),
            "image_data": encoded_string
        }
        response = app.test_client().post('/object_detection', json=payload)
        combined_results.append(response.get_json())

    return jsonify(combined_results)

def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=True)
