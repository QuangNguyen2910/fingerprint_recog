import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Union
import json
import uvicorn
from pyngrok import ngrok
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from utils.skeletonize import skeletonize
from utils.matching_fingerprint import load_minutiae_from_json, fingerprint_matching_ransac
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files at /static for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount sample_inputs directory to serve images
app.mount("/sample_inputs", StaticFiles(directory="sample_inputs"), name="sample_inputs")

# Create directories for converted images
os.makedirs("converted_uploaded", exist_ok=True)
os.makedirs("converted_matches", exist_ok=True)

# Mount directories for converted images
app.mount("/converted_uploaded", StaticFiles(directory="converted_uploaded"), name="converted_uploaded")
app.mount("/converted_matches", StaticFiles(directory="converted_matches"), name="converted_matches")

class MatchResult(BaseModel):
    fingerprint_id: Union[str, None]
    score: float
    image_path: Union[str, None]

class MatchResponse(BaseModel):
    top_matches: List[MatchResult]
    uploaded_image_path: Union[str, None]
    success: bool


def convert_tif_to_jpg(input_path: str, output_path: str) -> bool:
    """Convert a .tif image to .jpg and save it to the output path."""
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
    print("Normalization completed")
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    print("Segmentation completed")
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    print("Orientation calculation completed")
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    print("Frequency estimation completed")
    gabor_img = gabor_filter(normim, angles, freq)
    print("Gabor filtering completed")
    thin_image = skeletonize(gabor_img)
    print("Skeletonization completed")
    minutias, minutiae_list = calculate_minutiaes(thin_image)
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
        
        # Log the file type
        file_extension = file.filename.split('.')[-1].lower() if file.filename else "unknown"
        print(f"Uploaded image format: {file_extension}")
        print(f"Image loaded, shape: {img.shape}")

        # Save the uploaded image temporarily and convert if necessary
        temp_filename = f"{uuid.uuid4()}.{file_extension}"
        temp_path = f"converted_uploaded/{temp_filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        uploaded_image_path = f"/converted_uploaded/{temp_filename}"
        if file_extension == "tif":
            jpg_filename = f"{uuid.uuid4()}.jpg"
            jpg_path = f"converted_uploaded/{jpg_filename}"
            if convert_tif_to_jpg(temp_path, jpg_path):
                os.remove(temp_path)  # Remove the original .tif
                uploaded_image_path = f"/converted_uploaded/{jpg_filename}"
                print(f"Converted uploaded image to: {uploaded_image_path}")
            else:
                os.remove(temp_path)
                uploaded_image_path = None
                print("Failed to convert uploaded .tif to .jpg")
        else:
            print(f"Serving uploaded image as: {uploaded_image_path}")

        input_minutiae = fingerprint_pipeline(img)
        print(f"Extracted minutiae from uploaded image: {input_minutiae}")
        print(f"Number of minutiae points: {len(input_minutiae)}")
        minutiae_data = load_minutiae_from_json("minutiae_metadata.json")
        print(f"Loaded database with {len(minutiae_data)} entries")

        # Calculate scores for all fingerprints
        matches = []
        for fingerprint_id, minutiae_list in minutiae_data.items():
            score = fingerprint_matching_ransac(input_minutiae, minutiae_list)
            print(f"Comparing with {fingerprint_id}: Score = {score}")
            cp = 101
            cnt = int(fingerprint_id.split("_")[1])
            if cnt >= 10:
                cp += int(cnt / 10)
                cnt %= 10
            file_name = f"{cp}_{cnt}"
            # Try .tif
            current_dir = os.getcwd()
            print(current_dir)
            image_path = f"/sample_inputs/{file_name}.tif"
            image_disk_path = f"sample_inputs/{file_name}.tif"
            if os.path.exists(image_disk_path):
                # Convert .tif to .jpg
                converted_filename = f"{file_name}.jpg"
                converted_disk_path = f"converted_matches/{converted_filename}"
                if convert_tif_to_jpg(image_disk_path, converted_disk_path):
                    converted_image_path = f"/converted_matches/{converted_filename}"
                    print(f"Converted {fingerprint_id} to: {converted_image_path}")
                else:
                    print(f"Failed to convert {fingerprint_id} .tif to .jpg")
            else:
                image_path = None
                print(f"No image found for {fingerprint_id} (tried .jpg and .tif)")
                
            matches.append({"fingerprint_id": fingerprint_id, "score": score, "image_path": converted_image_path})

        # Sort matches by score in descending order and take top 3
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:3]

        # Convert to MatchResult objects
        top_matches = [MatchResult(fingerprint_id=m["fingerprint_id"], score=m["score"], image_path=m["image_path"])
                       for m in top_matches]

        return MatchResponse(
            top_matches=top_matches,
            uploaded_image_path=uploaded_image_path,
            success=len(top_matches) > 0 and top_matches[0].score > 0
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return MatchResponse(top_matches=[], uploaded_image_path=None, success=False)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    # ngrok.set_auth_token("2cepwyvVTQXCVosdf8pvOycUrJH_2sgV4QQUeAjCWmW2uYdee")
    # public_url = ngrok.connect(3636)
    # print(f"Public URL: {public_url}")
    uvicorn.run(app, host="localhost", port=3636)