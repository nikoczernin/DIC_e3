import boto3
from botocore.exceptions import NoCredentialsError



class AWS:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('yoloDB')

    def upload_to_aws(self, local_file, bucket, s3_file):
        try:
            self.s3.upload_file(local_file, bucket, s3_file)
            print(f"Upload Successful: {s3_file}")
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False


    def write_to_dynamodb(self, image_path, objects, confidences):
        try:
            response = self.table.put_item(
                Item={
                    'ImagePath': image_path,
                    'Objects': objects,
                    'ClassIDs': objects,
                    'Confidences': confidences
                }
            )
            print("Write to DynamoDB Successful")
        except Exception as e:
            print(f"Error writing to DynamoDB: {e}")



def main():
    # Define your local file, bucket name, and the S3 file name
    local_file_path = 'input_folder/000000000069.jpg'  # Replace with your image file path
    bucket = 'yolobuck'  # Your bucket name
    s3_file_path = '000000000069.jpg'  # Name you want to save as in S3

    # Initialize the AWS class
    aws = AWS()
    # Upload the file
    uploaded = aws.upload_to_aws(local_file_path, bucket, s3_file_path)



if __name__ == '__main__':
    main()