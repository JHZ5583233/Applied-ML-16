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
# Fake model for testing API

#class FakeModel:
    #def eval(self):
        #pass  # mimic the real model's eval method

    #def __call__(self, input_tensor):
        # input_tensor shape: (1, 3, H, W)
        #batch_size, _, height, width = input_tensor.shape
        # Return a fake depth map (e.g. 1 channel output)
        #random_tensor = torch.rand((batch_size, 3, height, width))
        #return random_tensor

# Use the fake model
#model = FakeModel()
#model.eval()


preprocessor = Preprocessing(tile_size=(256, 256))


@app.post("/predict_depth")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Read image as uint8
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(pil_image)  # Keep as uint8

        # Tile first, then process each tile
        tiles = preprocessor.tile_with_padding(img_array)

        depth_tiles = []
        for tile in tiles:
            # Convert to tensor WITH NORMALIZATION
            input_tensor = preprocessor.to_tensor(tile).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                depth_np = output.squeeze().cpu().numpy()
                depth_tiles.append(depth_np)

        # Reconstruct depth map
        depth_map = preprocessor.reconstruct_depth(depth_tiles)

        # Convert to RGB with proper scaling
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

