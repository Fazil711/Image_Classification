import io
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
from PIL import Image

app = FastAPI()

templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on: {device}")

try:
    model = torch.jit.load("cat_dog_portable_v3.pt", map_location=device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'cat_dog_portable.pt' is in the same folder!")

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@app.get("/")
def home(request: Request):
    """Serve the HTML homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return prediction."""
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    input_tensor = preprocess(image).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        output = model(input_tensor)

    print("\n--- DEBUG INFO ---")
    print(f"Raw Output Tensor: {output}") 
    print(f"Probabilities: {torch.nn.functional.softmax(output[0], dim=0)}")

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, predicted_idx = torch.max(output, 1)
    
    predicted_label = "Dog" if predicted_idx.item() == 1 else "Cat"
    confidence = probabilities[predicted_idx].item() * 100
    
    return {
        "filename": file.filename,
        "prediction": predicted_label,
        "confidence": f"{confidence:.2f}%"
    }