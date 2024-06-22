from pprint import pprint
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from time import sleep

# AWS class to handle interactions with S3 and DynamoDB
class AWS:
    def __init__(self):
        # Initialize S3 client
        self.s3 = boto3.client('s3')
        # Initialize DynamoDB resource with the specific region
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        # Reference the DynamoDB table named 'yoloDB'
        self.table = self.dynamodb.Table('yoloDB')


    # Method to upload a file to AWS S3
    def upload_to_s3(self, local_file, bucket, s3_file):
        # If no specific S3 file name is given, use the local file name
        if s3_file is None:
            s3_file = local_file.split("/")[-1]
        try:
            # Upload the file to the specified S3 bucket
            self.s3.upload_file(local_file, bucket, s3_file)
            print(f"Upload Successful: {s3_file}")
            return True
        except FileNotFoundError:
            # Handle the case where the local file is not found
            print("The file was not found")
            return False
        except NoCredentialsError:
            # Handle the case where AWS credentials are not available
            print("Credentials not available")
            return False


    # Method to write an item to DynamoDB
    def write_to_dynamodb(self, image_path, objects, confidences):
        try:
            # Put an item into the DynamoDB table
            response = self.table.put_item(
                Item={
                    'ImagePath': image_path,
                    'Objects': objects,
                    'ClassIDs': objects,  # This seems to be a duplicate, consider removing if not necessary
                    'Confidences': confidences
                }
            )
            print("Write to DynamoDB Successful")
        except Exception as e:
            # Handle any exceptions that occur during the DynamoDB put operation
            print(f"Error writing to DynamoDB: {e}")

    # Method to scan all items from the DynamoDB table
    def scan_dynamodb(self):
        try:
            # Scan the table and get all items
            response = self.table.scan()
            items = response.get('Items', [])
            return items
        except NoCredentialsError:
            # Handle the case where AWS credentials are not available
            return "Credentials not available"
        except PartialCredentialsError:
            # Handle the case where AWS credentials are incomplete
            return "Incomplete credentials provided"
        except Exception as e:
            # Handle any other exceptions that occur during the scan operation
            return f"Error reading from DynamoDB: {e}"

# Main function to demonstrate usage
def main():
    # Define your local file, bucket name, and the S3 file name
    local_file_path = 'input_folder/000000000448.jpg'  # Replace with your image file path
    bucket = 'yolobuck'  # Your bucket name
    s3_file_path = local_file_path.split("/")[-1]  # Name you want to save as in S3

    # Initialize the AWS class
    aws = AWS()
    # Upload the file to S3 (commented out for demonstration purposes)
    uploaded = aws.upload_to_s3(local_file_path, bucket, s3_file_path)
    sleep(5)
    # Scan the DynamoDB table and print the items
    pprint(aws.scan_dynamodb())


def db_contains_value(db, key, value):
    for obj in db:
        if obj[key] == value:
            return True
    return False


# Entry point of the script
if __name__ == '__main__':
    main()
