import json
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform

def load_minutiae_from_json(file_path):
    """
    Đọc dữ liệu minutiae từ file JSON.
    :param file_path: Đường dẫn tới file JSON chứa minutiae
    :return: Danh sách minutiae (list các dict)
    """
    with open(file_path, "r") as f:
        return json.load(f)

def match_minutiae_ransac(query_minutiae, db_minutiae):
    """
    So khớp minutiae giữa ảnh đầu vào và ảnh trong cơ sở dữ liệu bằng thuật toán RANSAC.
    :param query_minutiae: Danh sách minutiae của ảnh đầu vào (list các dict)
    :param db_minutiae: Danh sách minutiae của ảnh trong cơ sở dữ liệu (list các dict)
    :return: Số lượng minutiae khớp nhau sau khi áp dụng RANSAC
    """
    # Nếu số lượng minutiae nhỏ hơn 3 thì không đủ để so khớp
    if len(query_minutiae) < 3 or len(db_minutiae) < 3:
        return 0

    # Chuyển danh sách minutiae thành mảng numpy dạng [x, y]
    src_pts = np.array([(m["x"], m["y"]) for m in query_minutiae])
    dst_pts = np.array([(m["x"], m["y"]) for m in db_minutiae])

    # Áp dụng thuật toán RANSAC để tìm phép biến đổi affine tốt nhất giữa hai tập điểm
    model, inliers = ransac(
        (src_pts, dst_pts),
        AffineTransform,
        min_samples=3,
        residual_threshold=10,
        max_trials=1000
    )

    # Trả về số lượng điểm khớp (inliers), nếu không có thì trả về 0
    return int(np.sum(inliers)) if inliers is not None else 0
