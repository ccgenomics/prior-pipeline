import boto3
import botocore

class DownloadService:
    def download_file(bucket_name, key, local_name):
        s3 = boto3.resource('s3')
        try:
            s3.Bucket(bucket_name).download_file(key, local_name)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
