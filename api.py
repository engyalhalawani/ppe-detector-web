import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse, HTMLResponse
import os
from ppe_detector import load_ppe_model, draw_detections, add_stats_panel, normalize_class_name

# Global reference to the model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model on startup and clean up on shutdown."""
    global model
    print("⏳ [Startup] Loading PPE detection model...")
    try:
        model = load_ppe_model()
        print("✅ [Startup] Model loaded successfully.")
    except Exception as e:
        print(f"❌ [Startup] Failed to load model: {e}")
        raise e
    yield
    print("🛑 [Shutdown] Cleaning up API resources...")

app = FastAPI(
    title="PPE Safety Detection API",
    description="FastAPI service to detect Personal Protective Equipment (PPE) like Helmets, Safety Vests, and Workers in images.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", summary="Health Check and API Metadata")
async def root():
    """Returns the health status of the API, whether the model is loaded, and available class labels."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded yet."}
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": list(model.names.values())
    }

@app.get("/ui", summary="Web UI for PPE Detection", response_class=HTMLResponse)
async def serve_ui():
    """Serves the user-friendly web interface."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/detect", summary="Detect PPE objects and return JSON metadata")
async def detect(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, etc.)"),
    conf_threshold: float = Query(0.30, ge=0.0, le=1.0, description="Confidence threshold for filtering detections")
):
    """
    Upload an image to detect safety gear.
    Returns details on class names, confidences, bounding boxes, and object tallies in a JSON payload.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image format.")
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Could not parse or decode the uploaded image file.")
        
    # Scale down if extremely large to save processing power and memory
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
    # YOLO Inference
    results = model(img, verbose=False)
    
    detections = []
    stats = {
        "Helmet": 0,
        "No Helmet": 0,
        "Safety Vest": 0,
        "No Vest": 0,
        "Person": 0,
    }
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_name = model.names.get(cls_id, "Unknown")
            label = normalize_class_name(raw_name)
            
            detections.append({
                "class": label,
                "confidence": round(conf, 4),
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })
            
            # Increment tallies
            for key in stats:
                if key.lower() in label.lower():
                    stats[key] += 1
                    break
                    
    return {
        "filename": file.filename,
        "width": img.shape[1],
        "height": img.shape[0],
        "detections_count": len(detections),
        "detections": detections,
        "stats": stats
    }

@app.post("/detect-image", summary="Detect PPE objects and return the annotated image")
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, etc.)"),
    conf_threshold: float = Query(0.30, ge=0.0, le=1.0, description="Confidence threshold for filtering detections")
):
    """
    Upload an image to detect safety gear.
    Returns the annotated image as a file stream (image/jpeg) with drawn bounding boxes and a dark stats panel at the bottom.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image format.")
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Could not parse or decode the uploaded image file.")
        
    # Scale down if extremely large
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
    # YOLO Inference
    results = model(img, verbose=False)
    
    # Draw boxes & compile stats
    annotated, stats = draw_detections(img, results, model.names, conf_threshold)
    
    # Draw stats ribbon at the bottom
    final_img = add_stats_panel(annotated, stats, file.filename)
    
    # Encode back to JPEG format
    success, encoded_img = cv2.imencode(".jpg", final_img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode the annotated image.")
        
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
