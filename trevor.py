import os
import random
import shutil
from glob import glob

def select_random_images(input_dir, output_dir, num_images=500):
    """
    Chọn ngẫu nhiên num_images ảnh từ thư mục input_dir và lưu vào output_dir.
    
    Args:
        input_dir (str): Đường dẫn đến thư mục chứa ảnh gốc
        output_dir (str): Đường dẫn đến thư mục lưu ảnh được chọn
        num_images (int): Số lượng ảnh cần chọn
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lấy danh sách tất cả file ảnh trong thư mục
    image_paths = glob(os.path.join(input_dir, "*.[tT][iI][fF][fF]")) + \
                  glob(os.path.join(input_dir, "*.[pP][nN][gG]")) + \
                  glob(os.path.join(input_dir, "*.[jJ][pP][gG]"))
    
    if not image_paths:
        raise ValueError(f"Không tìm thấy ảnh nào trong thư mục: {input_dir}")
    
    if len(image_paths) < num_images:
        raise ValueError(f"Thư mục chỉ có {len(image_paths)} ảnh, không đủ để chọn {num_images} ảnh")
    
    # Chọn ngẫu nhiên 500 ảnh
    selected_paths = random.sample(image_paths, num_images)
    
    # Sao chép ảnh vào thư mục mới
    for img_path in selected_paths:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        try:
            shutil.copy2(img_path, output_path)  # Sao chép giữ nguyên metadata
            print(f"Đã sao chép: {output_path}")
        except Exception as e:
            print(f"Cảnh báo: Không thể sao chép {img_path}: {e}")
    
    print(f"Hoàn tất! Đã sao chép {len(selected_paths)} ảnh vào {output_dir}")

# Cấu hình thư mục
input_directory = "data_resized"  # Thư mục chứa 6000 ảnh
output_directory = "data_selected"  # Thư mục lưu 500 ảnh được chọn
num_images_to_select = 500  # Số lượng ảnh cần chọn

# Chạy hàm
try:
    select_random_images(input_directory, output_directory, num_images_to_select)
except Exception as e:
    print(f"Lỗi: {e}")