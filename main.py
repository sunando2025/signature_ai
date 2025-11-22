import os
import json
import base64
from io import BytesIO
from typing import Tuple, OrderedDict
from uuid import uuid4
from collections import OrderedDict

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# --- Image Processing & Model Dependencies ---
# These require scikit-image (skimage) and pytorch (torch)
from skimage import io, transform, filters, img_as_ubyte
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- FASTAPI SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Unified Signature Verification")

# Assumes a 'templates' directory exists for Jinja2 rendering (e.g., for index.html)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- IMAGE PROCESSING UTILITIES ---

def normalize_image(img: np.ndarray,
                    canvas_size: Tuple[int, int] = (840, 1360)) -> np.ndarray:
    """
    Crops the signature tightly to its bounding box, cleans noise, and centers it 
    on a fixed-size white canvas for standardized input size.
    """
    blur_radius = 2
    # Apply Gaussian blur for smoother edges before thresholding
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)
    
    # Determine foreground/background threshold using Otsu's method
    try:
        threshold = filters.threshold_otsu(img)
    except ValueError:
        return np.ones(canvas_size, dtype=np.uint8) * 255
        
    binarized_image = blurred_image > threshold
    
    # Find bounding box
    r, c = np.where(binarized_image == 0)
    if r.size == 0 or c.size == 0:
        return np.ones(canvas_size, dtype=np.uint8) * 255

    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    cropped = img[r.min(): r.max(), c.min(): c.max()]
    
    img_rows, img_cols = cropped.shape
    max_rows, max_cols = canvas_size

    # Calculate starting point to center the image on the canvas
    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center

    # Handle overflow and centering adjustments
    if img_rows > max_rows:
        r_start = 0
        difference = img_rows - max_rows
        crop_start = difference // 2
        cropped = cropped[crop_start:crop_start + max_rows, :]
        img_rows = max_rows
    else:
        if (r_start + img_rows) > max_rows: r_start -= ((r_start + img_rows) - max_rows)
        if r_start < 0: r_start = 0

    if img_cols > max_cols:
        c_start = 0
        difference = img_cols - max_cols
        crop_start = difference // 2
        cropped = cropped[:, crop_start:crop_start + max_cols]
        img_cols = max_cols
    else:
        if (c_start + img_cols) > max_cols: c_start -= ((c_start + img_cols) - max_cols)
        if c_start < 0: c_start = 0

    # Place the cropped signature onto the white canvas
    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    normalized_image[r_start:r_start + img_rows, c_start:c_start + img_cols] = cropped
    
    # Re-apply threshold for clean background
    normalized_image[normalized_image > threshold] = 255

    return normalized_image

def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Scales the image while maintaining aspect ratio, then center crops to target size."""
    height, width = size
    width_ratio = float(img.shape[1]) / width
    height_ratio = float(img.shape[0]) / height

    # Choose the ratio that results in the smallest dimension hitting the target size
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(img.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(img.shape[0] / width_ratio))

    img = transform.resize(img, (resize_height, resize_width),
                           mode='constant', anti_aliasing=True, preserve_range=True)

    img = img.astype(np.uint8)
    
    # Center crop after resizing
    if width_ratio > height_ratio:
        start = int(round((resize_width - width) / 2.0))
        return img[:, start:start + width]
    else:
        start = int(round((resize_height - height) / 2.0))
        return img[start:start + height, :]

def crop_center(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Helper to crop the exact center of an image."""
    img_shape = img.shape
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    return img[start_y: start_y + size[0], start_x:start_x + size[1]]

def preprocess_signature(img: np.ndarray,
                         canvas_size: Tuple[int, int],
                         img_size: Tuple[int, int],
                         input_size: Tuple[int, int] =(150, 220)) -> np.ndarray:
    """Main preprocessing pipeline."""
    img = img.astype(np.uint8)
    
    # 1. Normalize/Center
    centered = normalize_image(img, canvas_size)
    
    # 2. Invert colors (signature becomes white on black background)
    inverted = 255 - centered
    
    # 3. Resize and Center Crop
    resized = resize_image(inverted, img_size)

    # 4. Final crop to model's input size (150x220)
    if input_size is not None and input_size != img_size:
        cropped = crop_center(resized, input_size)
    else:
        cropped = resized

    return cropped

def decode_image(data_uri: str) -> np.ndarray:
    """Decode a base64 Data URI string to a grayscale NumPy array."""
    try:
        _, encoded = data_uri.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image = io.imread(BytesIO(binary_data), as_gray=True)
        return img_as_ubyte(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image data: {str(e)}. Data URI structure invalid.")


# --- PYTORCH MODEL DEFINITIONS (FIXED FOR NAMING MISMATCH) ---

class CustomBlock(nn.Module):
    """
    A unified block for (Conv/Linear) -> BN -> Mish activation.
    This structure ensures the layer names align with the saved state_dict.
    The keys in the state_dict look for 'conv' or 'fc' and 'bn' inside the named block.
    """
    def __init__(self, in_features, out_features, kernel_size=None, stride=1, pad=0, is_linear=False):
        super().__init__()
        if is_linear:
            # Linear layers
            self.fc = nn.Linear(in_features, out_features, bias=False)
            self.bn = nn.BatchNorm1d(out_features)
        else:
            # Convolutional layers
            self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, pad, bias=False)
            self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.Mish()

    def forward(self, x):
        if hasattr(self, 'conv'):
            x = self.conv(x)
        else:
            x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SigNet(nn.Module):
    """
    The base convolutional network for feature extraction (Siamese sub-network).
    Uses OrderedDict to explicitly name layers like 'conv1', 'conv2', etc., 
    which resolves the PyTorch loading error.
    """
    def __init__(self):
        super(SigNet, self).__init__()
        self.feature_space_size = 2048 # Output dimension
        
        # Convolutional layers using named blocks
        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', CustomBlock(1, 96, 11, stride=4)),
            ('pool1', nn.MaxPool2d(3, 2)),
            ('conv2', CustomBlock(96, 256, 5, pad=2)),
            ('pool2', nn.MaxPool2d(3, 2)),
            ('conv3', CustomBlock(256, 384, 3, pad=1)),
            ('conv4', CustomBlock(384, 384, 3, pad=1)),
            ('conv5', CustomBlock(384, 256, 3, pad=1)),
            ('pool3', nn.MaxPool2d(3, 2))
        ]))
        
        # Fully connected layers using named blocks
        self.fc_layers = nn.Sequential(OrderedDict([
            # Input size is 256 channels * 3 rows * 5 columns after the last pooling layer on a 150x220 input
            ('fc1', CustomBlock(256 * 3 * 5, 2048, is_linear=True)),
            ('fc2', CustomBlock(self.feature_space_size, self.feature_space_size, is_linear=True))
        ]))

    def forward_once(self, img):
        """Processes a single signature image to produce an embedding."""
        x = self.conv_layers(img)
        x = x.view(x.shape[0], -1) # Flatten
        x = self.fc_layers(x)
        return x

    def forward(self, img1, img2):
        """Standard forward method."""
        # Reshape and normalize to [0, 1] range
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2

class SiameseModel(nn.Module):
    """The complete Siamese Network structure with final projection and classification heads."""
    def __init__(self):
        super(SiameseModel, self).__init__()
        self.model = SigNet()
        
        # --- Model Loading ---
        MODEL_PATH = os.path.join(BASE_DIR, "single_model_old.pth")
        
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}. Verification will use uninitialized weights.")
        else:
            try:
                state_dict_tuple = torch.load(MODEL_PATH, map_location='cpu')
                # Assumes state dictionary is the first element in the saved tuple
                state_dict = state_dict_tuple[0]
                
                # Load the state dictionary into the SigNet sub-module
                self.model.load_state_dict(state_dict)
                print("SignatureNet weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights (This should now be resolved if the file is correct): {e}") 

        # Final layers for comparison/classification
        self.probs = nn.Linear(4, 1) 
        self.projection2d = nn.Linear(self.model.feature_space_size, 2) 

    def forward_once(self, img):
        return self.model.forward_once(img)

    def forward(self, img1, img2):
        # Normalize and reshape the input tensors
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)
        
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)

        # Project embeddings into 2D space 
        embedding1_2d = self.projection2d(embedding1)
        embedding2_2d = self.projection2d(embedding2)
        
        # Concatenate 2D embeddings for the final classification layer
        output = torch.cat([embedding1_2d, embedding2_2d], dim=1)
        output = self.probs(output) 

        return embedding1_2d, embedding2_2d, output
    
# Global Model Initialization
try:
    VERIFICATION_MODEL = SiameseModel()
    VERIFICATION_MODEL.eval()
    print("PyTorch Signature Model initialized and set to evaluation mode.")
except Exception as e:
    VERIFICATION_MODEL = None
    print(f"FATAL: Failed to initialize PyTorch model. Check dependencies and model file path: {e}")
    
# --- API Data Schema ---

class VerificationRequest(BaseModel):
    image1: str  # Base64 data URI of the reference signature
    image2: str  # Base64 data URI of the test signature

# --- FASTAPI ENDPOINTS ---

@app.get("/", response_class=HTMLResponse, summary="Main UI Page")
async def serve_ui(request: Request):
    """Renders the main frontend UI page. Assumes index.html exists in a 'templates' directory."""
    context = {
        "request": request,
        "app_name": "AI Signature Verifier",
        "user_name": "FastAPI/PyTorch Demo" 
    }
    return templates.TemplateResponse("index.html", context)

@app.post("/verify_signature", response_model=dict, summary="Verify Signature Authenticity")
async def verify_signature(data: VerificationRequest):
    """
    Receives two base64 images, processes them, and returns the classification result.
    """
    if VERIFICATION_MODEL is None:
        raise HTTPException(status_code=503, detail="Model service is currently unavailable. Check server logs for model initialization errors.")
    
    try:
        # 1. Decode and Preprocess Images
        img1_array = decode_image(data.image1)
        img2_array = decode_image(data.image2)
        
        # Preprocessing parameters 
        canvas_size = (952, 1360) 
        img_size = (256, 256) 
        
        img1_processed = preprocess_signature(img1_array, canvas_size, img_size)
        img2_processed = preprocess_signature(img2_array, canvas_size, img_size)

        # 2. Convert to Tensor
        img1_tensor = torch.tensor(img1_processed).unsqueeze(0)
        img2_tensor = torch.tensor(img2_processed).unsqueeze(0)

        # 3. Model Inference
        with torch.no_grad():
            output1, output2, confidence_raw = VERIFICATION_MODEL(img1_tensor, img2_tensor)
            
            # Convert raw logit to probability
            confidence = torch.sigmoid(confidence_raw).item() 
            
            # Calculate Cosine Similarity
            cos_sim = F.cosine_similarity(F.normalize(output1), F.normalize(output2)).item()
            
            cos_sim = max(0, cos_sim) 
            
        # 4. Classification Decision
        classification_threshold = 0.5 
        classification = 'Genuine' if confidence > classification_threshold else 'Forged'

        return JSONResponse({
            'similarity': f"{cos_sim * 100:.2f}%", 
            'confidence': confidence,
            'classification': classification
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected Verification Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {type(e).__name__}: {str(e)}")


if __name__ == '__main__':
    # Starts the FastAPI server
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)