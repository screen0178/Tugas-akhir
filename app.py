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

# Create the FastAPI app
app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client["image_service"]  # Database name
images_collection = db["images"]  # Collection name

# Create directories for storing images
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/processed", exist_ok=True)

# cors
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
        "public_url": public_url,
        "processed_url": None,
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
        output, _ = upsampler.enhance(img, outscale=4)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=f"Super-resolution failed: {error}")
    
    # Save the processed image
    processed_path = f"static/processed/{image_id}.jpg"
    cv2.imwrite(processed_path, output)
    
    # Generate a public URL for the processed image
    processed_url = f"/static/processed/{image_id}.jpg"
    
    # Update metadata in MongoDB
    images_collection.update_one(
        {"image_id": image_id},
        {"$set": {"processed_path": processed_path, "processed_url": processed_url}},
    )
    
    # Return the image ID and processed image URL
    return JSONResponse({
        "image_id": image_id,
        "processed_url": processed_url,
    })

@app.get("/api/result-download")
async def result_download(image_id: str):
    # Retrieve metadata from MongoDB
    image_metadata = images_collection.find_one({"image_id": image_id})
    if not image_metadata:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Check if the processed image exists
    processed_path = image_metadata.get("processed_path")
    if not processed_path or not os.path.exists(processed_path):
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    # Return the processed image file
    return FileResponse(processed_path, media_type="image/jpeg")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)