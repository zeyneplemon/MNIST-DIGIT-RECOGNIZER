# 🔢 Digit Recognizer
A real-time handwritten digit recognizer app built with a Convolutional Neural Network (CNN) trained with the MNIST dataset.

Draw any digit (0-9) on the canvas on the left and the model will predict it instantly!

<img width="800" height="431" alt="ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/10d0c816-b4d6-4618-985a-d1f67f209573" />

---

# 🚀 Demo
Draw a digit on the sketchpad -> hit Submit -> get the prediction.

Built with TensorFlow/Keras for the model and Gradio for interactive UI.


# 🧠 Model Architecture

The model is a CNN trained for 10 epochs on the MNIST dataset (60,000 training images):

```
Input (28x28x1)
  → Conv2D(32, 3x3, ReLU)
  → MaxPooling2D(2x2)
  → Dropout(0.25)
  → Conv2D(64, 3x3, ReLU)
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.5)
  → Dense(10, Softmax)
```

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Test Accuracy: ~99%

## Training Curve

<img width="1272" height="974" alt="1776265471319_image" src="https://github.com/user-attachments/assets/27054718-a59c-4db3-81d7-83ffdeb9f4af" />

Train and test loss converge closely — no significant overfitting thanks to **Dropout regularization.**


# 🛠️ How It Works

1. User draws a digit on the Gradio Sketchpad
2. The image is converted to grayscale and inverted
3. The digit is cropped to its bounding box and centered on a 28×28 canvas (matching MNIST format)
4. The image is normalized and passed through the CNN
5. The predicted digit is displayed as output


# 📦 Installation

```git clone https://github.com/YOUR_USERNAME/digit-recognizer.git
cd digit-recognizer
pip install -r requirements.txt
python app.py
```

# 📋 Requirements

```tensorflow
gradio
numpy
Pillow
```
Install all by typing:

```pip install -r requirements.txt
```

# 📁 Project Structure
```
digit-recognizer/
│
├── app.py              # Model + Gradio UI
├── requirements.txt    # Dependencies
├── loss_curve.png      # Training/test loss plot
└── README.md
```
# 🔮 Future Improvements
* Save and download the trained model to avoid retraining on each run.
* Add confidence scores for top 3 predictions.
* Deploy Hugging Face Spaces.
* Support multi-digit recignition.
  
---
# 👩‍💻 Author
**Zeynep Ünal**
[Linkedin](https://www.linkedin.com/in/zeynep-ünal-zeynepunal/) - [Github](https://github.com/zeyneplemon)

