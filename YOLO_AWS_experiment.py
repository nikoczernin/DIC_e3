# use the AWS Api to send 100 images to our YOLO lambda function
# and track the time it takes

import json
from time import sleep, time
from AWS_api import AWS
import os
from pprint import pprint
import pandas as pd

# Main function to process all images in the input folder and send them to the endpoint
def upload_100(input_folder):
    """
    Uploads 100 images from the specified input folder to an S3 bucket and logs the upload time.

    Args:
    input_folder (str): Path to the folder containing the images to be uploaded.

    Returns:
    None: The function writes the upload data to a JSON file.
    """
    aws = AWS()
    bucket = "yolobuck"  # Your S3 bucket name

    # List to store the results of each upload, including transfer time and file size
    upload_data = []

    # Loop through all image files in the input folder
    for image_filename in os.listdir(input_folder):
        s3_file_path = image_filename.split("/")[-1]  # Name to save as in S3
        image_path = os.path.join(input_folder, image_filename)

        # Upload the image to S3 and measure the time taken
        start_time = time()
        uploaded = aws.upload_to_s3(image_path, bucket, s3_file_path)
        end_time = time()
        transfer_time = end_time - start_time

        # Append the result to the upload_data list
        upload_data.append({
            "Filename": image_filename,
            "TransferTime": transfer_time,
            "FileSize": os.path.getsize(image_path)
        })

    print("100 uploads done")

    # Write the upload data to a JSON file
    with open("output/aws/uploaded_aws_data.json", "w") as f:
        json.dump(upload_data, f)




# Main function to process all images in the input folder and send them to the endpoint
def upload_1000(input_folder):
    """
    Uploads 100 images from the specified input folder to an S3 bucket and logs the upload time.

    Args:
    input_folder (str): Path to the folder containing the images to be uploaded.

    Returns:
    None: The function writes the upload data to a JSON file.
    """
    aws = AWS()
    bucket = "yolobuck"  # Your S3 bucket name

    # List to store the results of each upload, including transfer time and file size
    upload_data = []

    # Loop through all image files in the input folder
    for image_filename in os.listdir(input_folder):
        # do every image 1000 times
        for i in range(100):


            s3_file_path = image_filename.split("/")[-1] + f"_{i}" # Name to save as in S3
            image_path = os.path.join(input_folder, image_filename)

            # Upload the image to S3 and measure the time taken
            start_time = time()
            uploaded = aws.upload_to_s3(image_path, bucket, s3_file_path)
            end_time = time()
            transfer_time = end_time - start_time

            # Append the result to the upload_data list
            upload_data.append({
                "Filename": image_filename + f"_{i}",
                "TransferTime": transfer_time,
                "FileSize": os.path.getsize(image_path)
            })

    print("1000 uploads done")

    # Write the upload data to a JSON file
    with open("output/aws/uploaded_aws_data.json", "w") as f:
        json.dump(upload_data, f)




def test_single(local_file_path='input_folder/000000000554.jpg'):
    """
    Test function to upload a single image to S3 and print the DynamoDB scan results.
    """
    bucket = 'yolobuck'  # Your bucket name
    s3_file_path = local_file_path.split("/")[-1]  # Name you want to save as in S3

    # Initialize the AWS class
    aws = AWS()

    # Upload the file to S3
    uploaded = aws.upload_to_s3(local_file_path, bucket, s3_file_path)
    sleep(5)  # Wait for 5 seconds to ensure the upload completes

    # Scan the DynamoDB table and print the items
    pprint(aws.scan_dynamodb())






def get_data():
    """
    Scans the DynamoDB table and returns the items.

    Args:
    None

    Returns:
    list: A list of items from the DynamoDB table.
    """
    # Initialize the AWS class
    aws = AWS()
    return aws.scan_dynamodb()



def run_experiment(EXPERIMENT=100):

    if EXPERIMENT == 100:
        upload_100("input_folder")
    elif EXPERIMENT == 1000
        upload_1000("input_folder")

    sleep(20)  # Wait for 20 seconds to ensure all uploads and processing are done

    res = get_data()
    if isinstance(res, str):
        if res.startswith("Error"):
            raise Exception(res + "\nCheck credentials in ~/.aws/credentials")

    # Convert numerical values in the response to float
    for item in res:
        item['Confidences'] = [float(conf) for conf in item['Confidences']]
        item['InferenceTime'] = float(item['InferenceTime'])

    # Create a DataFrame from the response and drop the 'filename' column
    res = pd.DataFrame(res).drop(["filename"], axis=1)

    # Remove entries without a filename (these are considered old)
    res = res[res.Filename.notna()]

    # Filter out duplicate entries based on the 'Filename' column, keeping the last occurrence
    res = res.drop_duplicates(subset="Filename", keep="last")

    # Load the upload data from the JSON file
    with open("output/aws/uploaded_aws_data.json", "r") as f:
        upload_data = json.load(f)

    # Create a DataFrame from the upload data
    upload_data = pd.DataFrame(upload_data)
    print(upload_data.columns)

    # Merge the results DataFrame with the upload data DataFrame on the 'Filename' column
    final = res.merge(upload_data, on="Filename", how="inner")
    print(final.columns)

    # Write the final DataFrame to a CSV file
    final.to_csv(f"output/aws/aws_experiment_results_{EXPERIMENT}.csv", )





def main():
    # set this number to 100 or 1000 to run the corresponding experiment
    EXPERIMENT = 100
    run_experiment(EXPERIMENT)

    # or alternatively only test a single image
    #test_single()



if __name__ == '__main__':
    main()