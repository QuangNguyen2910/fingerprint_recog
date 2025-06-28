import cv2 as cv
from glob import glob
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from utils.skeletonize import skeletonize
from utils.matching_fingerprint import load_minutiae_from_json, fingerprint_matching_ransac


def fingerprint_pipline(input_img):
    block_size = 16

    normalized_img = normalize(input_img.copy(), float(100), float(100))

    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    gabor_img = gabor_filter(normim, angles, freq)

    thin_image = skeletonize(gabor_img)

    minutias, minutiae_list  = calculate_minutiaes(thin_image)

    minutiae_dict = [{"x": m[0], "y": m[1], "type": m[2]} for m in minutiae_list]
    return minutiae_dict

if __name__ == '__main__':
    img_dir = './input_test/*'
    output_dir = './output/'
    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path,0) for img_path in images_paths])
    
    minutiae_data = load_minutiae_from_json("minutiae_data.json")
    input_image = cv.imread("input_test/OIP.jpg", 0)
    input_minutiae = fingerprint_pipline(input_image)
    best_match = None
    best_score = 0

    for fingerprint_id, minutiae_list in minutiae_data.items():
        score = fingerprint_matching_ransac(input_minutiae, minutiae_list)
        if score > best_score:
            best_score = score
            best_match = fingerprint_id

    if best_match:
        print(f"✅ Ảnh đầu vào khớp với {best_match} ({best_score} điểm khớp)")
    else:
        print("❌ Không tìm thấy ảnh vân tay nào khớp")
