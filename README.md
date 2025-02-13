# 🖼️ Image Captioning Using Deep Learning  

## 📌 Overview  
This project implements an **Image Captioning Model** using a combination of **CNNs and LSTMs**. The model extracts features from images using a **Convolutional Neural Network (CNN)** and generates textual descriptions using **Long Short-Term Memory (LSTM) networks**. It lies at the intersection of **Computer Vision and Natural Language Processing (NLP)**.

## 🚀 Features  
- **CNN (ResNet/VGG16/DenseNet)** for image feature extraction  
- **LSTM for text generation** based on extracted image features  
- **Tokenization and embedding layers** for processing captions  
- **Attention mechanism** to improve caption accuracy  
- **BLEU Score Evaluation** for assessing model performance  

## 📂 Dataset  
- **Dataset Used:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
- **Preprocessing:**  
  - Convert text to lowercase  
  - Remove special characters, numbers, and extra spaces  
  - Add `<start>` and `<end>` tags to captions for sequence modeling  

## 🏗️ Model Architecture  
### **1️⃣ Encoder - CNN**
- Pretrained models like **VGG16, ResNet50, or DenseNet** are used.
- Extracts image feature vectors.

### **2️⃣ Decoder - LSTM**
- Converts image feature vectors into text using **word embeddings**.
- Uses **LSTM units** to generate sequential captions.

### **3️⃣ Text Tokenization & Embedding**
- Captions are tokenized and converted into **word embeddings**.
- Uses a **lookup table** to map words to vector representations.
