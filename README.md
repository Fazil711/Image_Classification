🐶 Cat vs. Dog Classifier
A Full-Stack AI Web Application that classifies images of Cats and Dogs with high accuracy using a custom Convolutional Neural Network (CNN). Trained on the Kaggle Dataset and deployed using FastAPI.

📸 Demo
Tip: Add a screenshot of your website here to show off the UI!

🚀 Features
Deep Learning Model: Custom CNN trained on 25,000 images using PyTorch.

Fast Backend: Powered by FastAPI for high-performance inference.

Interactive UI: Clean HTML/CSS frontend with a real-time confidence progress bar.

Portable: Uses TorchScript (.pt) for easy deployment without heavyweight dependencies.

GPU Support: Auto-detects CUDA to speed up inference if available.

🛠️ Tech Stack
Model Training: Google Colab (T4 GPU), PyTorch, Torchvision

Backend: FastAPI, Uvicorn

Frontend: HTML5, CSS3, JavaScript (Fetch API)

Dataset: Kaggle Dog & Cat Classification Dataset

📂 Project Structure
Bash

cat_dog_web_app/
├── 📂 templates/
│   └── index.html          # The Frontend UI
├── cat_dog_final.pt        # The Trained Model (TorchScript)
├── main.py                 # FastAPI Backend Server
├── requirements.txt        # Dependencies
└── README.md               # This file
⚡ Installation & Setup
1. Clone the Repository
Bash

git clone https://github.com/yourusername/cat-dog-classifier.git
cd cat-dog-classifier
2. Create a Virtual Environment (Optional but Recommended)
Bash

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install Dependencies
Bash

pip install torch torchvision fastapi uvicorn python-multipart pillow
4. Run the App
Bash

uvicorn main:app --reload
5. Open in Browser
Go to: http://127.0.0.1:8000

🧠 Model Training Details
The model was trained on Google Colab using a standard CNN architecture:

Input: 128x128 RGB Images

Architecture: 3 Convolutional Layers + Max Pooling + 2 Fully Connected Layers

Data Augmentation: Resize, Normalization

Accuracy: ~81% on Test Set (5 Epochs)

If you want to re-train the model yourself, check out the training_notebook.ipynb (optional: upload your colab file).

🔮 Future Improvements
[ ] Add support for more animal classes.

[ ] Deploy to Render or HuggingFace Spaces.

[ ] Improve accuracy with ResNet-18 Transfer Learning.

🤝 Contributing
Feel free to fork this repo and submit pull requests!

📜 License
This project is open-source and available under the MIT License.