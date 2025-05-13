from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO
from backend.train import AnimalCNN

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from all origins (you can restrict it to specific origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify the frontend's domain like ['http://localhost:3000']
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define class names (replace with your actual class labels)cd..
CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# Load the trained model (first define the architecture and then load the state_dict)
def load_model():
    model = AnimalCNN(num_classes=len(CLASS_NAMES))  # Create the model instance
    model.load_state_dict(torch.load('model/animal_cnn.pth'))  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model

# Load the model once at startup
model = load_model()

# Define the transformation to preprocess the image (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')  # Open the image using PIL

        # Apply transformation to the image (resize and convert to tensor)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Make a prediction with the model
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            class_id = predicted.item()  # Convert to Python integer
            class_name = CLASS_NAMES[class_id]  # Get the class name

        # Return the prediction as a JSON response
        return JSONResponse(content={"prediction": class_name})

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Start the FastAPI app if this file is run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#uvicorn app:app --reload
