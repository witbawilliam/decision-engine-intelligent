import logging
import os
from storage.s3_client import S3Client

# Setup logging
logging.basicConfig(level=logging.INFO)

def verify():
    # Bypass get_settings() by passing values directly to the constructor
    # These match the 'aws configure' values you used earlier
    client = S3Client(
        bucket_name="ml-datasets",
        endpoint_url="http://127.0.0.1:9000",
        access_key="12345",
        secret_key="your_secret_key", # Replace with your actual secret
        region="us-east-1"
    )
    
    print(f"--- S3 Health Check ---")
    if client.ping():
        print(" S3 Connection: SUCCESS")
    else:
        print(" S3 Connection: FAILED (Check if MinIO is running on port 9000)")
        return

    prefix = "datasets/witba_123/"
    print(f"\nScanning prefix: {prefix}...")
    
    objects = client.list_objects(prefix=prefix)
    
    if objects:
        print(f"Found {len(objects)} files:")
        for obj in objects:
            print(f" - {obj}")
    else:
        print(f" No files found. Check if the bucket name 'ml-datasets' is correct.")

if __name__ == "__main__":
    verify()