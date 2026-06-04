# Flower Image Classification using CNN

A deep learning application for classifying flower images using Convolutional Neural Networks (CNN) with a beautiful Streamlit web interface.

## 🌸 Features

- **5 Flower Classification**: Daisy, Dandelion, Rose, Sunflower, Tulip
- **Deep Learning Model**: CNN-based architecture trained on flower datasets
- **Streamlit Interface**: Interactive web application for real-time predictions
- **Confidence Scores**: View prediction confidence for all flower classes
- **Image Upload**: Support for JPG, JPEG, and PNG formats
- **Responsive Design**: Beautiful UI with custom styling

## 📋 Requirements

- Python 3.8+
- TensorFlow >= 2.13.0
- Streamlit >= 1.28.1
- NumPy >= 1.24.0
- Pillow >= 10.0.0

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/connecthassan/Flower-image-classification-using-cnn.git
cd Flower-image-classification-using-cnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Flower-image-classification-using-cnn/
├── app.py                              # Main Streamlit application
├── classification.ipynb                # Training notebook with CNN model
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── models/
│   └── flower_classification_model.h5  # Trained CNN model
└── data/
    └── [flower_images]                 # Training/test datasets
```

## 🎯 How to Use

1. **Upload an Image**: Click on the upload button to select a flower image
2. **View Prediction**: The model will analyze and display the predicted flower type
3. **Check Confidence**: See confidence scores for all flower classes
4. **Adjust Threshold**: Use the sidebar slider to adjust prediction confidence threshold

## 🧠 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels (RGB)
- **Output Classes**: 5 flower types
- **Training Framework**: TensorFlow/Keras

## 📊 Supported Flower Classes

- 🌼 **Daisy** - A white flowering plant
- 🌻 **Dandelion** - Yellow flowering weed
- 🌹 **Rose** - Classic red/pink flower
- 🌻 **Sunflower** - Large yellow flower
- 🌷 **Tulip** - Spring flowering bulb

## 📝 License

This project is open source and available for educational and research purposes.

## 👨‍💻 Author

**Connect Hassan**
- GitHub: [@connecthassan](https://github.com/connecthassan)

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests.

## 📧 Contact

For questions or suggestions, please reach out through GitHub issues.

---

Built with ❤️ using TensorFlow, Keras, and Streamlit
