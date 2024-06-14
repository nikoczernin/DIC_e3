from flask import Flask, request, jsonify  # Flask backend
import cv2  # OpenCV image processing
import numpy as np  # Numpy handling arrays
import base64  # Base64 encoding/decoding images
import uuid  # UUID generating unique identifiers
import os  # OS  file operations
from python.yolo.Yolo import Yolo  # Yolo class object detection

# Initialize the Flask application
app = Flask(__name__)

# Initialize the Yolo object with specified thresholds for confidence and NMS
yolo = Yolo(confidence_threshold=0.0, nms_threshold=0.0)

# Route for object detection
@app.route('/object_detection', methods=['POST'])
def object_detection():
    # Parse the incoming JSON request
    data = request.json
    image_id = data.get('id')
    image_data = data.get('image_data')

    try:
        # Decode the base64 image data to a numpy array
        image_np = _decode_image(image_data)
        # Save the decoded image to a temporary path
        img_path = _save_image(image_np, image_id)

        # By commenting out the lines, boxes can be drawn around the elements in addition to recognising the objects.
        # This can be used to better understand the performance and accuracy of the model
        # Perform object recognition on the saved image and draw boxes with identified objects
        #image, (class_ids, confidences, boxes) = yolo.transform_draw(img_path)
        # Perform object recognition on the saved image without draw boxes
        object_detection_time, class_ids, confidences, boxes = yolo.transform_and_time(img_path)

        # Format the results
        detected_objects = _format_detection_results(class_ids, confidences)
        # Prepare the response
        response = {
            "id": image_id,
            "objects": detected_objects,
            "inference_time": object_detection_time
        }
        # Return response as JSON
        return jsonify(response)
    except Exception as e:
        # Error handling
        return jsonify({"error": str(e)}), 400
    finally:
        # Clean up the saved image file
        if os.path.exists(img_path):
            # Image removal can be enabled if necessary
            os.remove(img_path)
            #pass

# Decode base64 image data to a numpy array
def _decode_image(image_data):
    image_bytes = base64.b64decode(image_data)
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decoding failed")
    return img

# Save the numpy array as an image file
def _save_image(img, image_id):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, f"{image_id}.jpg")
    cv2.imwrite(img_path, img)
    return img_path

# Format the object detection results
def _format_detection_results(class_ids, confidences):
    detected_objects = []
    for class_id, confidence in zip(class_ids, confidences):
        detected_object = {
            "label": yolo.classes[class_id],
            "accuracy": confidence
        }
        detected_objects.append(detected_object)
    return detected_objects

# Route for testing the local execution of object detection
@app.route('/test', methods=['GET'])
def test_local_execution():
    input_folder = "input_folder"
    combined_results = []

    # Loop through all files in the input folder
    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)
        if not os.path.isfile(image_path):
            print(f"No file: {image_path}")
            continue
        # Encode the image to base64
        encoded_string = _encode_image(image_path)
        # Prepare the payload for the POST request
        payload = {
            "id": str(uuid.uuid4()),
            "image_data": encoded_string
        }
        # POST request to object detection route
        response = app.test_client().post('/object_detection', json=payload)
        combined_results.append(response.get_json())

    # Return results as JSON
    return jsonify(combined_results)

# Encode an image file to base64
def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Run Flask application
if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=True)
