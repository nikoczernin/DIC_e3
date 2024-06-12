import boto3
from botocore.exceptions import NoCredentialsError

# Initialize a session using Amazon S3
s3 = boto3.client('s3')

def upload_to_aws(local_file, bucket, s3_file):
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print(f"Upload Successful: {s3_file}")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def main():
    # Define your local file, bucket name, and the S3 file name
    local_file_path = 'path_to_your_image.jpg'  # Replace with your image file path
    bucket = 'yolobuck'  # Your bucket name
    s3_file_path = 'your_image.jpg'  # Name you want to save as in S3

    # Upload the file
    uploaded = upload_to_aws(local_file_path, bucket, s3_file_path)

if __name__ == '__main__':
    main()