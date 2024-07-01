import requests  # HTTP requests
import base64  # Encoding images to base64
import uuid  # Eenerating unique identifiers
import os  # File operations
import sys  # Command-line arguments
import time  # Measuring time intervals
import csv  # Writing CSV files

# Encode an image to a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Send a POST request with encoded image to endpoint
def post_image(image_path, endpoint):
    encoded_string = encode_image(image_path)
    payload = {
        # Generate a unique ID for the image
        "id": str(uuid.uuid4()),
        # base64 encoded image data
        "image_data": encoded_string
    }
    # Set content type JSON
    headers = {'Content-Type': 'application/json'}

    try:
        # Start time for the transfer
        start_transfer_time = time.time()
        # Send POST request
        response = requests.post(endpoint, json=payload, headers=headers)
        # End time for the transfer
        end_transfer_time = time.time()

        # Calculate transfer time
        transfer_time = end_transfer_time - start_transfer_time

        # Print HTTP status code
        print(f"Status Code: {response.status_code}")
        # Print response content
        print(f"Response Content: {response.content.decode('utf-8')}")
        # Raise exception for HTTP errors
        response.raise_for_status()

        # Return response data, transfer time and inference time
        return response.json(), transfer_time

    except Exception as err:
        # Print error
        print(f"Error: {err}")
        # Return error details
        return {"error": str(err)}, 0, 0

# Process and save the responses
def process_responses(combined_results):
    # Output text file results
    output_file = "output/output.txt"
    # CSV file only timing logs
    timelog_file = "output/timelog.csv"

    # Write results to the output file
    with (open(output_file, "w") as f):
        for result in combined_results:
            data = result.get("data", {})
            transfer_time = result.get("transfer_time", 0)
            inference_time = data.get('inference_time')

            image_id = data.get('id', 'unknown')
            objects = data.get('objects', [])
            f.write(f"Image ID: {image_id}\n")
            for obj in objects:
                label = obj.get('label')
                accuracy = obj.get('accuracy')
                f.write(f"Detected: {label} with accuracy: {accuracy}\n")
            f.write(f"Transfer Time: {transfer_time} seconds\n")
            f.write(f"Inference Time: {inference_time} seconds\n")
            f.write("-" * 42 + "\n")

    # Write timing logs to the CSV file
    with open(timelog_file, "w", newline='') as csvfile:
        fieldnames = ['image_id', 'transfer_time', 'inference_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in combined_results:
            data = result.get("data", {})
            transfer_time = result.get("transfer_time", 0)
            inference_time = data.get('inference_time')

            image_id = data.get('id', 'unknown')
            writer.writerow({
                'image_id': image_id,
                'transfer_time': transfer_time,
                'inference_time': inference_time
            })

# Main function to process all images in the input folder and send them to the endpoint
def main(input_folder, endpoint):
    combined_results = []

    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)
        if not os.path.isfile(image_path):
            print(f"No file: {image_path}")
            continue

        # Send image and store results
        result, transfer_time = post_image(image_path, endpoint)
        combined_results.append({
            "data": result,
            "transfer_time": transfer_time
        })
    # Process and save results
    process_responses(combined_results)

# Run in terminal: python client.py input_folder http://127.0.0.1:5000/object_detection
if __name__ == '__main__':
    # Print usage if incorrect arguments
    if len(sys.argv) != 3:
        print("Correct usage: python client.py <input_folder> <endpoint>")
        sys.exit(1)

    input_folder = sys.argv[1]
    endpoint = sys.argv[2]

    # Run the main function
    main(input_folder, endpoint)

    # Execute the main function 100x for the experiment
    #for _ in range(100):
    #    main(input_folder, endpoint)
