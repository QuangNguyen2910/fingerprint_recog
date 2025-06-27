import os
import uuid
import cv2 as cv
import numpy as np
from typing import List, Union

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from utils.skeletonize import skeletonize
from utils.matching_fingerprint import load_minutiae_from_json, fingerprint_matching_ransac

# --- Constants and Directory Setup ---
STATIC_DIR = "static"
SAMPLE_INPUTS_DIR = "sample_inputs"
CONVERTED_UPLOADED_DIR = "converted_uploaded"
CONVERTED_MATCHES_DIR = "converted_matches"

os.makedirs(CONVERTED_UPLOADED_DIR, exist_ok=True)
os.makedirs(CONVERTED_MATCHES_DIR, exist_ok=True)

# --- FastAPI App and Middleware ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/sample_inputs", StaticFiles(directory=SAMPLE_INPUTS_DIR), name="sample_inputs")
app.mount("/converted_uploaded", StaticFiles(directory=CONVERTED_UPLOADED_DIR), name="converted_uploaded")
app.mount("/converted_matches", StaticFiles(directory=CONVERTED_MATCHES_DIR), name="converted_matches")

# --- Pydantic Models ---
class MatchResult(BaseModel):
    fingerprint_id: Union[str, None]
    score: float
    image_path: Union[str, None]

class MatchResponse(BaseModel):
    top_matches: List[MatchResult]
    uploaded_image_path: Union[str, None]
    success: bool

def convert_tif_to_png(input_path: str, output_path: str) -> bool:
    """Convert a .tif image to .png and save it."""
    img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image for conversion: {input_path}")
        return False
    success = cv.imwrite(output_path, img)
    if not success:
        print(f"Failed to save converted image: {output_path}")
    return success

def fingerprint_pipeline(input_img):
    block_size = 16
    
    normalized_img = normalize(input_img.copy(), float(100), float(100))
    
    _, normim, mask = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    
    gabor_img = gabor_filter(normim, angles, freq)
    
    thin_image = skeletonize(gabor_img)
    
    _, minutiae_list = calculate_minutiaes(thin_image)
    minutiae_dict = [{"x": m[0], "y": m[1], "type": m[2]} for m in minutiae_list]

    print("Minutiae extraction completed")

    return minutiae_dict

@app.post("/upload-fingerprint", response_model=MatchResponse)
async def upload_fingerprint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("Error: Could not decode image")
            return MatchResponse(top_matches=[], uploaded_image_path=None, success=False)

        print(f"Image loaded, shape: {img.shape}")

        input_minutiae = fingerprint_pipeline(img)

        minutiae_data = load_minutiae_from_json("minutiae_feature.json")
        print(f"Loaded database with {len(minutiae_data)} entries")

        matches = []
        for fingerprint_id, minutiae_list in minutiae_data.items():
            score = fingerprint_matching_ransac(input_minutiae, minutiae_list)
            print(f"Score for {fingerprint_id}: {score}")

            cp = 101
            cnt = int(fingerprint_id.split("_")[1])
            if cnt >= 10:
                cp += int(cnt / 10)
                cnt %= 10
            file_name = f"{cp}_{cnt}"
            tif_path = os.path.join(SAMPLE_INPUTS_DIR, f"{file_name}.tif")

            matches.append({
                "fingerprint_id": fingerprint_id,
                "score": score,
                "tif_path": tif_path,
                "image_path": None  
            })

        # Sắp xếp và lấy top 3
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches_raw = matches[:3]

        for match in top_matches_raw:
            tif_path = match["tif_path"]
            if os.path.exists(tif_path):
                png_filename = os.path.basename(tif_path).replace(".tif", ".png")
                png_path_disk = os.path.join(CONVERTED_MATCHES_DIR, png_filename)
                if convert_tif_to_png(tif_path, png_path_disk):
                    match["image_path"] = f"/{CONVERTED_MATCHES_DIR}/{png_filename}"
                    print(f"Converted top match to PNG: {match['image_path']}")
                else:
                    print(f"Failed to convert .tif for {match['fingerprint_id']}")

        top_matches = [
            MatchResult(
                fingerprint_id=m["fingerprint_id"],
                score=m["score"],
                image_path=m["image_path"]
            ) for m in top_matches_raw
        ]

        return MatchResponse(
            top_matches=top_matches,
            uploaded_image_path=None,
            success=len(top_matches) > 0 and top_matches[0].score > 0
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return MatchResponse(top_matches=[], uploaded_image_path=None, success=False)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3636)