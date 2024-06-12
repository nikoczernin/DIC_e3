import json
import cv2
import numpy as np
import boto3
from Yolo import Yolo

s3_client = boto3.client('s3')


def lambda_handler(event, context):
    message = f"so, nun zu dir mein lieber {np.random.randint(1, 400)}!"
    print(message)
    # Get the bucket name and object key from the event
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    print(bucket_name)
    object_key = event['Records'][0]['s3']['object']['key']
    print(object_key)

    # Download the image file from S3
    download_path = f'/tmp/{object_key}'
    s3_client.download_file(bucket_name, object_key, download_path)

    # create the Yolo transformer
    yolo = Yolo()

    # transform the received image
    image_transformed = yolo.transform_draw(download_path)

    # save the transformed image
    transformed_path = f'/tmp/transformed-{object_key}'
    yolo.save_image(image_transformed, transformed_path)

    # upload the edited image
    print("Uploading to", f'transformed/transformed-{object_key}', "in bucket", bucket_name)
    s3_client.upload_file(transformed_path, bucket_name, f'transformed/transformed-{object_key}')

    return {
        'statusCode': 200,
        'body': json.dumps(message)
    }
