import requests
import base64
import uuid
import json
import os
import sys
import time
import csv


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
        start_transfer_time = time.time()
        response = requests.post(endpoint, json=payload, headers=headers)
        end_transfer_time = time.time()

        transfer_time = end_transfer_time - start_transfer_time

        print(f"Status Code: {response.status_code}")
        print(f"Response Content: {response.content.decode('utf-8')}")
        response.raise_for_status()  # Raise an exception for HTTP errors

        inference_time = response.elapsed.total_seconds()

        return response.json(), transfer_time, inference_time
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": str(http_err)}, 0, 0
    except Exception as err:
        print(f"Other error occurred: {err}")
        return {"error": str(err)}, 0, 0


def process_responses(combined_results):
    output_file = "output/output_100.txt"
    timelog_file = "output/timelog_100.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for result in combined_results:
            data = result.get("data", {})
            transfer_time = result.get("transfer_time", 0)
            inference_time = result.get("inference_time", 0)

            image_id = data.get('id', 'unknown')
            objects = data.get('objects', [])
            f.write(f"Image ID: {image_id}\n")
            for obj in objects:
                label = obj.get('label')
                accuracy = obj.get('accuracy')
                f.write(f"Detected: {label} with accuracy: {accuracy}\n")
            f.write(f"Transfer Time: {transfer_time} seconds\n")
            f.write(f"Inference Time: {inference_time} seconds\n")
            f.write("-" * 30 + "\n")

    with open(timelog_file, "w", newline='') as csvfile:
        fieldnames = ['image_id', 'transfer_time', 'inference_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in combined_results:
            data = result.get("data", {})
            transfer_time = result.get("transfer_time", 0)
            inference_time = result.get("inference_time", 0)

            image_id = data.get('id', 'unknown')
            writer.writerow({
                'image_id': image_id,
                'transfer_time': transfer_time,
                'inference_time': inference_time
            })


def main(input_folder, endpoint):
    combined_results = []

    for _ in range(100):
        for image_filename in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_filename)
            if not os.path.isfile(image_path):
                print(f"No file: {image_path}")
                continue

            result, transfer_time, inference_time = post_image(image_path, endpoint)
            combined_results.append({
                "data": result,
                "transfer_time": transfer_time,
                "inference_time": inference_time
            })

    process_responses(combined_results)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python client.py <input_folder> <endpoint>")
        sys.exit(1)

    input_folder = sys.argv[1]
    endpoint = sys.argv[2]

    main(input_folder, endpoint)
