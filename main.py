from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class ImageResponse(BaseModel):
    image: str  # Base64 encoded image with background removed
    face_count: int  # Number of faces detected in the image

@app.post("/remove-background/", response_model=ImageResponse)
async def remove_background(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        
        # Convert to PIL Image
        input_image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL image to numpy array for face detection
         
      
        
        # Remove background
        output_image = remove(input_image)
        
        # Convert back to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return ImageResponse(image=img_str)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
