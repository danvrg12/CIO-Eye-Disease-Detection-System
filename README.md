# Eye Disease Detection System 🩺

A machine learning-powered web application that detects various eye diseases from uploaded images using deep learning and Flask.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Features](#features)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Description

This project uses transfer learning with MobileNetV2 to classify eye diseases from images. The system can detect 5 different eye conditions: Bulging Eyes, Cataracts, Crossed Eyes, Glaucoma, and Uveitis. Built with Flask for easy deployment and a simple web interface for image uploads.

**Problem it solves**: Provides an accessible tool for preliminary eye disease screening using computer vision, helping to identify potential eye conditions that may require professional medical attention.

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd eye-disease-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install flask tensorflow pillow numpy
```

4. **Create project structure**
```bash
mkdir model dataset dataset/train templates
```

5. **Organize your dataset**
```
dataset/
└── train/
    ├── Bulging_Eyes/
    ├── Cataracts/
    ├── Crossed_Eyes/
    ├── Glaucoma/
    └── Uveitis/
```

## Usage

1. **Train the model** (first time only)
```bash
python train_model.py
```

2. **Start the Flask application**
```bash
python app.py
```

3. **Access the web interface**
   - Open your browser to `http://localhost:5000`
   - Upload an eye image
   - Get instant prediction results

## Examples

### Web Interface Usage
1. Navigate to `http://localhost:5000`
2. Click "Choose File" and select an eye image
3. Click "Upload" to get prediction
4. View results showing detected condition

### API Usage
```bash
curl -X POST \
  http://localhost:5000/predict \
  -F "image_file=@path/to/eye_image.jpg"
```

**Response:**
```json
{
  "prediction": "Cataracts"
}
```

## Features

- **🤖 AI-Powered Detection** - Uses MobileNetV2 transfer learning
- **🌐 Web Interface** - Simple upload and prediction interface
- **📱 REST API** - JSON-based prediction endpoint
- **⚡ Fast Processing** - Optimized for quick inference (~1-2 seconds)
- **🔄 Automatic Preprocessing** - Handles image resizing and normalization
- **📊 Multiple Conditions** - Detects 5 different eye diseases
- **💾 Lightweight Model** - ~9MB model size for easy deployment

## API Reference

### Endpoints

#### `GET /`
Serves the main web interface.

#### `POST /predict`
Predicts eye disease from uploaded image.

**Parameters:**
- `image_file` (file): Image file to analyze

**Response:**
```json
{
  "prediction": "string"  // One of: Bulging_Eyes, Cataracts, Crossed_Eyes, Glaucoma, Uveitis
}
```

**Error Response:**
```json
{
  "error": "string"  // Error description
}
```

### Model Specifications
- **Input Size**: 224x224x3
- **Architecture**: MobileNetV2 + Custom Classification Head
- **Classes**: 5 eye disease categories
- **Output**: Softmax probabilities

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Guidelines
- Follow PEP 8 for Python code
- Add comments for complex logic
- Test with various image types
- Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- **TensorFlow/Keras** - Deep learning framework
- **MobileNetV2** - Pre-trained model architecture
- **Flask** - Web application framework
- **PIL/Pillow** - Image processing library
- **Medical Community** - For eye disease classification standards

---

**⚠️ Medical Disclaimer**: This application is for educational and research purposes only. It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and proper diagnosis.
