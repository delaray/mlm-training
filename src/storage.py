from google.cloud import storage
import os
import warnings

"""
This module handles communicating with the Google Cloud bucket.
A local file can be downloaded or uploaded.

The GOOGLE_APPLICATION_CREDENTIALS env var must be properly initialized.

Additionally, the ENV_MODE env var must be set to either SANDBOX, DEV or PROD,
in order to access the appropriate folder in the bucket. Advanced functions
ensure data_available and upload_path enforce this by requiring blobs to be
located in a dir named the same as ENV_MODE.
"""

# From https://cloud.google.com/storage/docs/downloading-objects


def get_env_mode():
    """Checks that ENV_MODE is properly set and returns its value."""
    env_mode = os.environ.get("ENV_MODE")

    if env_mode is None or (env_mode != "SANDBOX" and env_mode != "DEV" and env_mode != "PROD"):
        raise Exception(
            "Environment variable ENV_MODE must be set to SANDBOX, DEV or PROD.")
    else:
        return env_mode


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    # Extract directory and create it if not exists already
    directory = os.path.dirname(destination_file_name)
    if directory:
        os.makedirs(directory, exist_ok=True)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Downloaded storage object {} from bucket {} to local file {}.".format(
        source_blob_name, bucket_name, destination_file_name)
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(
        source_file_name, destination_blob_name))


def list_blob_names(bucket_name):
    """Lists all the blobs in the bucket."""

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    return [blob.name for blob in blobs]


def ensure_data_available(bucket_name: str,
                          destination_path: str,
                          enforce_env_mode: bool = True,
                          overwrite_local: bool = False):
    """
    destination_path can be a file or dir: file if a single file corresponds
    to the blob prefix, dir if multiple files corresponding to this prefix.
    The ENV_MODE prefix (SANDBOX, DEV or PROD) is enforced by default, i.e. the
    bucket blob name must start with the correct prefix. To disable this check,
    set enforce_env_mode to False.
    """

    all_blobs = list_blob_names(bucket_name)
    source_blob_prefix = os.path.basename(destination_path)

    if enforce_env_mode:
        env_mode = get_env_mode()
        source_blob_prefix = env_mode + '/' + source_blob_prefix

    prefix_blobs = [blob_name for blob_name in all_blobs if blob_name.startswith(
        source_blob_prefix)]

    if len(prefix_blobs) == 0:
        raise Exception(
            f"Error: no blob name starts with {source_blob_prefix} in bucket {bucket_name}.")

    if len(prefix_blobs) == 1:
        if overwrite_local or not os.path.isfile(destination_path):
            download_blob(bucket_name, prefix_blobs[0], destination_path)

    else:
        # assuming destination_path is a dir
        os.makedirs(destination_path, exist_ok=True)
        for blob in prefix_blobs:
            filename = blob[len(source_blob_prefix)+1:]
            full_name = os.path.join(destination_path, filename)
            if overwrite_local or not os.path.isfile(full_name):
                download_blob(bucket_name, blob, full_name)


def upload_path(bucket_name, source_path, enforce_env_mode=True):
    """
    source_path can be a file or dir: if a single file, stored in the bucket
    under the file name; if a dir, the files inside are stored as dir/<file> in the bucket.
    No recursion applied: dirs inside the dir are ignored.
    The ENV_MODE prefix (SANDBOX, DEV or PROD) is enforced by default, i.e. the bucket blob
    name must start with the correct prefix. To disable this check, set enforce_env_mode to
    False (a warning will be printed).
    """

    source_blob_prefix = os.path.basename(source_path)

    if enforce_env_mode:
        env_mode = get_env_mode()
        source_blob_prefix = env_mode + '/' + source_blob_prefix
    else:
        warnings.warn(
            "Not enforcing ENV_MODE (SANDBOX, DEV or PROD) when writing to the Google bucket.")

    if os.path.isfile(source_path):
        upload_blob(bucket_name, source_path, source_blob_prefix)
    elif os.path.isdir(source_path):
        files = [f for f in os.listdir(source_path) if os.path.isfile(
            os.path.join(source_path, f))]
        for f in files:
            upload_blob(bucket_name, os.path.join(
                source_path, f), source_blob_prefix + '/' + f)
    elif not os.path.exists(source_path):
        raise Exception(f"Error: no such file or directory at {source_path}")
    else:
        raise Exception(f"Error: cannot determine the type of {source_path}")
