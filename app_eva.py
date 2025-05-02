from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import os
import uuid
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import time
from scipy.ndimage import zoom  # For bicubic interpolation
from skimage.metrics import structural_similarity as ssim  # For SSIM calculation

# Create the FastAPI app
app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client["image_service_eva"]  # Database name
images_collection = db["images"]  # Collection name

# Create directories for storing images
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/processed", exist_ok=True)
os.makedirs("static/bicubic", exist_ok=True)  # Directory for bicubic upscaled images

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for public URLs)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the RealESRGAN upsampler
def initialize_upsampler():
    model_name = "RealESRGAN_x4plus"
    model_path = os.path.join("weights", model_name + ".pth")
    netscale = 4
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=0,
    )
    return upsampler

upsampler = initialize_upsampler()

# Function to calculate PSNR
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    # Generate a unique ID for the image
    image_id = str(uuid.uuid4())
    
    # Save the uploaded image
    file_path = f"static/uploads/{image_id}.jpg"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Generate a public URL for the image
    public_url = f"/static/uploads/{image_id}.jpg"
    
    # Store metadata in MongoDB
    image_metadata = {
        "image_id": image_id,
        "original_path": file_path,
        "processed_path": None,
        "bicubic_path": None,
        "public_url": public_url,
        "processed_url": None,
        "bicubic_url": None,
        "processed_time": None,
        "bicubic_time": None,
        "psnr": None,
        "ssim": None,
    }
    images_collection.insert_one(image_metadata)
    
    # Return the image ID and public URL
    return JSONResponse({
        "image_id": image_id,
        "public_url": public_url,
    })

@app.post("/api/img-inference")
async def image_inference(image_id: str):
    # Retrieve metadata from MongoDB
    image_metadata = images_collection.find_one({"image_id": image_id})
    if not image_metadata:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Load the original image
    original_path = image_metadata["original_path"]
    img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    
    # Perform super-resolution
    try:
        start_time = time.time()
        output, _ = upsampler.enhance(img, outscale=4)
        end_time = time.time()
        elapsed_time = end_time - start_time
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=f"Super-resolution failed: {error}")
    
    # Save the processed image
    processed_path = f"static/processed/{image_id}.jpg"
    cv2.imwrite(processed_path, output)
    
    # Generate a public URL for the processed image
    processed_url = f"/static/processed/{image_id}.jpg"
    
    # Calculate PSNR and SSIM
    original_img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    processed_img = cv2.imread(processed_path, cv2.IMREAD_UNCHANGED)
    
    # Resize original image to match the processed image size
    original_img_resized = cv2.resize(original_img, (processed_img.shape[1], processed_img.shape[0]))
    
    psnr_value = calculate_psnr(original_img_resized, processed_img)
    ssim_value, _ = ssim(original_img_resized, processed_img, full=True, multichannel=True)
    
    # Update metadata in MongoDB
    images_collection.update_one(
        {"image_id": image_id},
        {"$set": {
            "processed_path": processed_path,
            "processed_url": processed_url,
            "processed_time": elapsed_time,
            "psnr": psnr_value,
            "ssim": ssim_value,
        }},
    )
    
    # Return the image ID, processed image URL, PSNR, and SSIM
    return JSONResponse({
        "image_id": image_id,
        "processed_url": processed_url,
        "processed_time": elapsed_time,
        "psnr": psnr_value,
        "ssim": ssim_value,
    })

@app.post("/api/bicubic-upscale")
async def bicubic_upscale(image_id: str, scale_factor: float = 2.0):
    # Retrieve metadata from MongoDB
    image_metadata = images_collection.find_one({"image_id": image_id})
    if not image_metadata:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Load the original image
    original_path = image_metadata["original_path"]
    img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    
    # Perform bicubic upscaling
    try:
        start_time = time.time()
        upscaled_img = zoom(img, (scale_factor, scale_factor, 1), order=3)  # Bicubic interpolation
        end_time = time.time()
        elapsed_time = end_time - start_time
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Bicubic upscaling failed: {error}")
    
    # Save