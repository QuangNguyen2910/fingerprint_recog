from utils.fingerprint_pipeline import fingerprint_pipline
from glob import glob
import numpy as np
import cv2 as cv
from tqdm import tqdm
import json


def open_images(directory):
    images_paths = glob(directory)
    return np.array([cv.imread(img_path,0) for img_path in images_paths])

if __name__ == '__main__':
    img_dir = './data_selected/*'
    output_dir = './output/'
    images = open_images(img_dir)
    # image pipeline
    # os.makedirs(output_dir, exist_ok=True)
    # print(images[0])
    # print(len(images))
    # minutiae_list = fingerprint_pipline(images[0])
    # Save minutiae data to json
    minutiae_database = {}

    for i, img in enumerate(tqdm(images)):
        fingerprint_id = f"pic_{i+1}"
        minutiae_dict = fingerprint_pipline(img)

        minutiae_database[fingerprint_id] = minutiae_dict

    with open("minutiae_metadata_fullfinger.json", "w") as f:
        json.dump(minutiae_database, f, indent=4)
    
    print("Save metadata succesfully !")