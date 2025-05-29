from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
from PIL import Image
import torch
import numpy as np
from project_name.models.cnn import CNNBackbone
from project_name.models.Preprocessing_class import Preprocessing

app = FastAPI()

# Load model
model = CNNBackbone(pretrained=False)
model.load_state_dict(torch.load("cnn_best.pth", map_location="cpu"))
model.eval()

preprocessor = Preprocessing(tile_size=(256, 256))

@app.post("/predict_depth")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Read image and tile
        contents = await file.read()
        #pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        #img_array = np.array(pil_image)
        img_array = preprocessor.load_image(io.BytesIO(contents))
        tiles = preprocessor.tile_with_padding(img_array)
        depth_tiles = []
        for tile in tiles:
            # Convert to tensor
            input_tensor = preprocessor.to_tensor(tile).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                depth_np = output.squeeze().cpu().numpy()
                depth_tiles.append(depth_np)

        # Reconstruct depth map and convert to RGB
        depth_map = preprocessor.reconstruct_depth(depth_tiles)
        depth_rgb = preprocessor.depth_to_rgb(depth_map, invert=True)

        # Create response
        result_image = Image.fromarray(depth_rgb)
        byte_io = io.BytesIO()
        result_image.save(byte_io, format="PNG")
        byte_io.seek(0)

        return StreamingResponse(byte_io, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing image: {str(e)}"}
        )

@app.get("/")
def read_root():
    return {"message": "API is running. Go to /docs to try it out."}

