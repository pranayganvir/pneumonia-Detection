# 🩺 Pneumonia Detection using Deep Learning

## 🔍 Overview

This project is a **Streamlit web application** that detects **pneumonia** from **X-ray images** using a pre-trained deep learning model. The application allows users to upload an X-ray image, processes it using a trained **TensorFlow/Keras model**, and provides a diagnosis result.

## ✨ Features

- 📂 **Upload X-ray Images** 🏥: Users can upload images in JPG, PNG, or JPEG formats.
- 🧠 **Deep Learning-Based Detection** 🤖: Uses a **trained CNN model** to classify the image.
- 🖼️ **Prediction Display on Image** 🖊️: The result is displayed directly on the uploaded image.
- 🎨 **Professional UI** 💻: Styled with Streamlit customizations for better user experience.

## ⚙️ Installation

### 📌 Prerequisites

- ✅ Make sure you have **Python 3.7+** installed.

### 🛠️ Steps

1. 📥 Clone the repository:
   ```bash
   git clone https://github.com/pranayganvir/pneumonia-Detection.git
   cd pneumonia-detector
   ```
2. 📦 Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. ▶️ Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🏥 Model

- 🏗️ The deep learning model is trained using **VGG19**, a pre-trained convolutional neural network model.
- 🔄 **Transfer learning** was applied to fine-tune the model for pneumonia detection.
- 💾 The model has been saved as `best_model.h5` and is loaded into the application for predictions.

**🔹 Note:** You can easily generate `best_model.h5` by training the model using the provided dataset and scripts.

- 🏗️ The deep learning model is trained using **VGG19**, a pre-trained convolutional neural network model.
- 🔄 **Transfer learning** was applied to fine-tune the model for pneumonia detection.
- 💾 The model has been saved as `best_model.h5` and is loaded into the application for predictions.

### 📊 Dataset

The model was trained using the **Chest X-Ray dataset**, which can be accessed at:
[🔗 Chest X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 📁 File Structure

```
📂 pneumonia-detector
│── 📜 app.py                # Main Streamlit app
│── 📦 best_model.h5         # Pre-trained CNN model
│── 📄 requirements.txt      # Dependencies
│── 📖 README.md             # Documentation
```

## 🛠️ Technologies Used

- 🐍 **Python** 🖥️
- 🔬 **TensorFlow/Keras** 🧪
- 🎭 **Streamlit** 🌐
- 🖼️ **PIL (Pillow) for image processing** 🏞️
- 📊 **NumPy** 🔢

## 🚀 Example Usage

1. 🔼 Open the Streamlit app.
2. 📤 Upload an X-ray image.
3. 🏥 The app will classify whether the image indicates pneumonia or not.
4. 🎨 The result will be displayed on the image.

## 🖼️ Screenshots

### 📌 Interface

![App Screenshot](https://github.com/pranayganvir/pneumonia-Detection/blob/main/Sceenshots/Screenshot%202025-03-12%20165103.png)

### 📌 Normal Image

![App Screenshot](https://github.com/pranayganvir/pneumonia-Detection/blob/main/Sceenshots/Screenshot%202025-03-12%20165042.png)

### 📌 Pneumonia Detected Image

![App Screenshot](https://github.com/pranayganvir/pneumonia-Detection/blob/main/Sceenshots/Screenshot%202025-03-12%20164955.png)








## 🔮 Future Improvements

- 🚀 Improve the model accuracy with more training data.
- 🧐 Add Grad-CAM visualization for interpretability.
- ☁️ Deploy on **Streamlit Cloud** or **AWS/GCP**.

## 👨‍💻 Author

- ✍️ **Pranay Ganvir**
- 🔗 **GitHub:** [pranayganvir](https://github.com/pranayganvir)

## 📜 License

This project is licensed under the **MIT License** 📄.

