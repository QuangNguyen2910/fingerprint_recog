import cv2 as cv
import numpy as np


def minutiae_at(pixels, i, j, kernel_size):

    if pixels[i][j] == 1:
        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1),  (1, 1), 
                     (1, 0),  (1, -1), (0, -1), (-1, -1)]  
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), 
                     (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0), 
                     (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]  

        values = [pixels[i + l][j + k] for k, l in cells]

        # Đếm số lần chuyển đổi từ 0 → 1 trên vòng tròn minutiae
        crossings = sum(abs(values[k] - values[k + 1]) for k in range(len(values) - 1)) // 2

        # Phân loại minutiae
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"

    return "none"


def calculate_minutiaes(im, kernel_size=3):
    """
    Tìm minutiae (điểm kết thúc và phân nhánh) và lưu danh sách đặc trưng.
    """
    binary_image = np.zeros_like(im)
    binary_image[im < 10] = 1 
    binary_image = binary_image.astype(np.int8)

    (height, width) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}

    minutiae_list = []

    for i in range(1, width - kernel_size // 2):
        for j in range(1, height - kernel_size // 2):
            minutiae = minutiae_at(binary_image, j, i, kernel_size)
            if minutiae != "none":
                cv.circle(result, (i, j), radius=2, color=colors[minutiae], thickness=2)
                minutiae_list.append((i, j, minutiae))  # Lưu vào danh sách

    return result, minutiae_list