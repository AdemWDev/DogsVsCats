# 🐶🐱 Dogs vs Cats Image Classifier

A deep learning project that classifies images as **dogs or cats** using **Convolutional Neural Networks (CNNs)** built with **PyTorch** and deployed through a **Gradio web app**.

---

## 📸 Demo
Try it locally:
```bash
python deploy.py
````

Then open the Gradio interface in your browser to upload an image.

---

## 🧠 Overview

This project demonstrates an end-to-end deep learning workflow:

* Data preprocessing and augmentation
* CNN model training and validation
* Model saving and loading for inference
* Gradio interface for deployment

---

## ⚙️ Project Structure

```
DogsVsCats/
├── data/                     # Dataset (train/test images)
├── training.py               # Model definition and training loop
├── deploy.py                 # Gradio app for inference
├── model.pth                 # Saved trained model
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🧩 Model Architecture

The current version uses a **custom CNN** with:

* 3 convolutional layers
* ReLU activations
* MaxPooling
* Fully connected layers for binary classification

> You can easily replace it with a pretrained network (e.g., VGG16 or ResNet18) for higher accuracy.

Example snippet:

```python
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(4096, 2)
```

---

## 📊 Training

1. Download the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Preprocess images (resize to 128x128)
3. Run training:

   ```bash
   python training.py
   ```
4. The best model weights are saved as `model.pth`

Training logs and accuracy plots can be added under `/logs/` or `/results/`.

---

## 🚀 Deployment

Launch the Gradio app:

```bash
python deploy.py
```

Upload an image — the model will predict **Dog 🐶** or **Cat 🐱**, along with its confidence score.

---

## 🧾 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical packages:

```
torch
torchvision
gradio
numpy
pillow
```

---

## 🧪 Example Predictions

| Image       | Prediction | Confidence |
| ----------- | ---------- | ---------- |
| 🐱 cat1.jpg | Cat 🐱     | 97.3%      |
| 🐶 dog2.jpg | Dog 🐶     | 92.5%      |

---

## 🧑‍💻 Author

**Adem Mhamdi**
📧 [am1.adem.mhamdi@gmail.com(mailto:am1.adem.mhamdi@gmail.com)]
💼 [LinkedIn]

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

```

---

Would you like me to **auto-generate the matching `requirements.txt` and `LICENSE` files** so your repo is instantly ready for GitHub upload?
```
