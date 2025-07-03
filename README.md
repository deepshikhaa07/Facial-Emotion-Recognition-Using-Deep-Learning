# Emotion Recognition using Deep Learning on FER2013 Dataset

We implemented and compared multiple deep learning models to perform facial emotion recognition using the FER2013 dataset. The best-performing model is integrated into a real-time application and we are working on a feature that overlays floating emojis based on the detected emotions.

## Models Implemented

1. **Basic CNN**
   - 3 convolutional blocks
   - ReLU activation, Softmax output
   - Trained from scratch

2. **ResNet50**
   - Pretrained on ImageNet
   - Fine-tuned with custom top layers

3. **MobileNetV2**
   - Lightweight model ideal for real-time
   - Efficient and fast

4. **EfficientNetB0**
   - Best accuracy on test set
   - Compound scaling with high performance

## Dataset

- **FER2013** (Facial Expression Recognition 2013)
- Grayscale 48x48 pixel images
- 7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Source: [Kaggle FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## Preprocessing

- Normalized pixel values
- Data augmentation:
  - Rotation (15°)
  - Zoom 0.2
  - Horizontal flip
    
## Training Configuration

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 64
- Epochs: 10


## Evaluation

- Confusion Matrix
- Classification Report
- Sample predictions with predicted labels


## Requirements

Install the required Python libraries using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python streamlit
