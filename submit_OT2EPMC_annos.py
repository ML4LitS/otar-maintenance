from minio import Minio
from minio.error import S3Error
import os
import time
from tqdm import tqdm

def upload_files_to_minio(bucket_name, source_dir, endpoint, access_key, secret_key):
    # Initialize the Minio client
    minio_client = Minio(endpoint,
                         access_key=access_key,
                         secret_key=secret_key,
                         secure=True)  # Use secure=True for HTTPS

    # Ensure the bucket exists, create it if it does not
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    # Get list of all .tar.gz files in the source directory
    files_to_upload = [f for f in os.listdir(source_dir) if f.endswith('.tar.gz')]
    
    # Iterate through all .tar.gz files with a progress bar
    for file_name in tqdm(files_to_upload, desc="Uploading files"):
        file_path = os.path.join(source_dir, file_name)
        try:
            # Upload the file
            with open(file_path, 'rb') as file_data:
                file_stat = os.stat(file_path)
                minio_client.put_object(bucket_name, file_name, file_data, file_stat.st_size, content_type="application/octet-stream")
            print(f"Successfully uploaded {file_name}.")
            time.sleep(1)  # Wait for 1 second after each upload
        except S3Error as e:
            print(f"Failed to upload {file_name}. Error: {e}")


# Example usage
endpoint = "annotations.europepmc.org"
access_key = "LRG1LM4D12K4ZTTOD8EC"
secret_key = "s6TjPu7fS15F35GcqeemgfWtuxr6UcVM2S2iwEHY"
bucket_name = "submissions"
source_dir = "/hps/nobackup/literature/otar-pipeline/quaterly_annotations_api/quaterly_latest"

upload_files_to_minio(bucket_name, source_dir, endpoint, access_key, secret_key)

