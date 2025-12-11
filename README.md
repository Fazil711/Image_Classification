# 🐶 Cat vs. Dog Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A Full-Stack AI Web Application that classifies images of Cats and Dogs with high accuracy using a custom Convolutional Neural Network (CNN). Trained on the Kaggle Dataset and deployed using FastAPI.

---

## 📸 Demo

> <img src="https://github.com/user-attachments/assets/e578b2e2-e7a5-4217-914b-dcc5585817ba" width="500" alt="Image Cat", height="700">
> <img width="500" height="700" alt="Image Dog" src="https://github.com/user-attachments/assets/cf94ea6c-22c4-4d22-9925-6934dd3faf1c" />
---

## 🚀 Features
* **Deep Learning Model:** Custom CNN trained on 25,000 images using PyTorch.
* **Fast Backend:** Powered by **FastAPI** for high-performance inference.
* **Interactive UI:** Clean HTML/CSS frontend with a real-time confidence progress bar.
* **Portable:** Uses TorchScript (`.pt`) for easy deployment without heavyweight dependencies.
* **GPU Support:** Auto-detects CUDA to speed up inference if available.

---

## 🛠️ Tech Stack
* **Model Training:** Google Colab (T4 GPU), PyTorch, Torchvision
* **Backend:** FastAPI, Uvicorn
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
* **Dataset:** [Kaggle Dog & Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

---

## 📂 Project Structure

```bash
cat_dog_web_app/
├── 📂 templates/
│   └── index.html
├── cat_dog_final.pt
├── main.py
├── requirements.txt
└── README.md
```

## ⚡ Installation & Setup
### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/cat-dog-classifier.git](https://github.com/yourusername/cat-dog-classifier.git)
cd cat-dog-classifier
```

### 2. Install Dependencies
```Bash
pip install torch torchvision fastapi uvicorn python-multipart pillow
```

### 3. Run the App
```Bash
uvicorn main:app --reload
```

### 4. Open in Browser
```
Go to: http://127.0.0.1:8000
```

## 🧠 Model Training Details
The model was trained on Google Colab using a standard CNN architecture:
* **Input:** 128x128 RGB Images
* **Architecture:** 3 Convolutional Layers + Max Pooling + 2 Fully Connected Layers
* **Data Augmentation:** Resize, Normalization
* **Accuracy:** ~81% on Test Set (5 Epochs)

---

## 🔮 Future Improvements
* [ ] Add support for more animal classes.
* [ ] Deploy to Render or HuggingFace Spaces.
* [ ] Improve accuracy with ResNet-18 Transfer Learning.

## 🤝 Contributing
Feel free to fork this repo and submit pull requests!

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).