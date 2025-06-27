import cv2 as cv
import os
import numpy as np
import json
from glob import glob

from utils.normalization import normalize
from utils.segmentation import create_segmented_and_variance_images
from utils.orientation import calculate_angles
from utils.frequency import ridge_freq
from utils.gabor_filter import gabor_filter
from utils.skeletonize import skeletonize
from utils.crossing_number import calculate_minutiaes


def open_images(dir):
    images_paths = sorted(glob(dir))  # ensure consistent order
    images = []
    for img_path in images_paths:
        img = cv.imread(img_path, 0)
        if img is not None:
            images.append((os.path.basename(img_path), img))
        else:
            print(f"Warning: Could not read image: {img_path}")
    return images


def pipeline(input_img):
    block_size = 16

    normalized_img = normalize(input_img.copy(), 100.0, 100.0)

    (_, normin, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    angles = calculate_angles(normalized_img, W=block_size)

    freq = ridge_freq(im=normin,
                      mask=mask,
                      orient=angles,
                      block_size=block_size,
                      kernel_size=5,
                      minWaveLength=5,
                      maxWaveLength=15)

    gabor_img = gabor_filter(im=normin,
                             orient=angles,
                             freq=freq)

    thin_img = skeletonize(gabor_img)

    minutias, minutiae_list = calculate_minutiaes(thin_img)

    minutiae_dict = [{"x": m[0], "y": m[1], "type": m[2]} for m in minutiae_list]

    return minutiae_dict, thin_img


def draw_minutiae(img, minutiae_list):

    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    for m in minutiae_list:
        x, y, typ = m["x"], m["y"], m["type"]
        color = (0, 255, 0) if typ == "ending" else (0, 0, 255)
        cv.circle(img_color, (x, y), 3, color, -1)
        cv.putText(img_color, typ[0], (x + 4, y - 4), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA)

    return img_color


if __name__ == '__main__':
    img_dir = './input_test/*'
    output_dir = './test_output/'

    os.makedirs(output_dir, exist_ok=True)
    images = open_images(img_dir)

    minutiae_database = {}

    for i, (img_name, img) in enumerate(images):
        img_id = f"img_{i + 1}"

        minutiae_dict, processed_img = pipeline(img)
        minutiae_database[img_id] = minutiae_dict

        annotated_img = draw_minutiae(img, minutiae_dict)

        output_path = os.path.join(output_dir, f"{img_id}_annotated.png")
        cv.imwrite(output_path, annotated_img)

    with open("minutiae_feature.json", "w") as f:
        json.dump(minutiae_database, f, indent=4)

    print("✅ SUCCESSFULLY SAVED MINUTIAE AND ANNOTATED IMAGES")
