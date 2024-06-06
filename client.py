import requests
import base64
import uuid
import json
import os
import sys

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def post_image(image_path, endpoint):
    encoded_string = encode_image(image_path)
    payload = {
        "id": str(uuid.uuid4()),
        "image_data": encoded_string
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response Content: {response.content.decode('utf-8')}")
        response.raise_for_status()  # Raise an exception for HTTP errors

        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": str(http_err)}
    except Exception as err:
        print(f"Other error occurred: {err}")
        return {"error": str(err)}

def process_responses(combined_results):
    output_file = "output.txt"
    with open(output_file, "w") as f:
        for result in combined_results:
            image_id = result.get('id', 'unknown')
            objects = result.get('objects', [])
            f.write(f"id: {image_id}\n")
            for obj in objects:
                label = obj.get('label')
                accuracy = obj.get('accuracy')
                f.write(f"label: {label} - accuracy: {accuracy}\n")
            f.write("-" * 20 + "\n")

def main(input_folder, endpoint):
    combined_results = []

    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)
        if not os.path.isfile(image_path):
            print(f"No file: {image_path}")
            continue

        result = post_image(image_path, endpoint)
        combined_results.append(result)

    process_responses(combined_results)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python client.py <input_folder> <endpoint>")
        sys.exit(1)

    input_folder = sys.argv[1]
    endpoint = sys.argv[2]

    main(input_folder, endpoint)
