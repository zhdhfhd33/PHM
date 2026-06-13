import os
import urllib.request
import zipfile

def download_and_extract(url, extract_to):
    zip_path = os.path.join(extract_to, "dataset.zip")
    
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download completed.")
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")
    
    # Clean up the zip file
    os.remove(zip_path)
    print("Clean up completed.")

if __name__ == "__main__":
    url = "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
    raw_dir = r"C:\Users\minkun\my_proj\PHM_ai_optimizatio\data\raw"
    os.makedirs(raw_dir, exist_ok=True)
    download_and_extract(url, raw_dir)
