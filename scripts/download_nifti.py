import os

import requests
from tqdm import tqdm


def sanitize_filename(filename):
    return "".join(c for c in filename if c.isalnum() or c in "._- ")

def download_series_nifti():
    base_url = "http://localhost:8042"
    auth = ("orthanc", "orthanc")
    output_dir = "nifti_files"

    os.makedirs(output_dir, exist_ok=True)

    try:
        series_url = f"{base_url}/series"
        series_response = requests.get(series_url, auth=auth)
        series_response.raise_for_status()
        series_ids = series_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching series list: {e}")
        return

    for i, series_id in tqdm(enumerate(series_ids)):
        if i >= 5:
            break

        try:
            series_metadata_url = f"{base_url}/series/{series_id}"
            metadata_response = requests.get(series_metadata_url, auth=auth)
            metadata_response.raise_for_status()
            metadata = metadata_response.json()

            modality = metadata["MainDicomTags"].get("Modality", "")
            protocol = metadata["MainDicomTags"].get("ProtocolName", "")
            series_num = metadata["MainDicomTags"].get("SeriesNumber", "")

            filename = sanitize_filename(f"{modality}_{protocol}_{series_num}_{series_id}.nii.gz")
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                print(f"File {filename} already exists, skipping...")
                continue

            nifti_url = f"{base_url}/series/{series_id}/nifti"
            nifti_response = requests.get(nifti_url, auth=auth)
            nifti_response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(nifti_response.content)

            print(f"Downloaded NIFTI for series {series_id}")

        except requests.exceptions.RequestException as e:
            print(f"Error processing series {series_id}: {e}")
            continue

if __name__ == "__main__":
    download_series_nifti()
