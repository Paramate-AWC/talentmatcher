from obs import ObsClient

def download_from_obs(access_key_id, secret_access_key, server, bucket_name, object_key, download_path):
    """
    Downloads an object from Huawei OBS to a local file.

    Parameters:
    - access_key_id (str): Your Huawei Cloud access key ID.
    - secret_access_key (str): Your Huawei Cloud secret access key.
    - server (str): The OBS endpoint URL for your region.
    - bucket_name (str): The name of the OBS bucket.
    - object_key (str): The key (path) of the object to download.
    - download_path (str): The local file path where the object will be saved.

    Returns:
    - None
    """
    # Initialize the OBS client
    obs_client = ObsClient(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        server=server
    )
    
    try:
        # Attempt to download the object
        resp = obs_client.getObject(bucket_name, object_key, downloadPath=download_path)
        if resp.status < 300:
            print(f'Object downloaded successfully to {download_path}')
        else:
            print(f'Error: {resp.errorMessage}')
    except Exception as e:
        print(f'An exception occurred: {e}')
    finally:
        # Close the OBS client to release resources
        obs_client.close()


if __name__ == "__main__":
    # Your Huawei Cloud credentials and OBS endpoint
    access_key_id = 'your-access-key-id'
    secret_access_key = 'your-secret-access-key'
    server = 'https://obs.your-region.myhuaweicloud.com'  # e.g., 'https://obs.cn-north-1.myhuaweicloud.com'

    # Parameters for the object to download
    bucket_name = 'your-bucket-name'
    object_key = 'path/to/your/object.ext'  # e.g., 'folder/file.txt'
    download_path = '/local/path/to/save/file.ext'

    # Call the function to download the object
    download_from_obs(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        server=server,
        bucket_name=bucket_name,
        object_key=object_key,
        download_path=download_path
    )
