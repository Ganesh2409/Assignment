# Image Captioning and Description Project -1 
## Technical Documentation

### Overview
This project is a web-based application that uses the BLIP (Bootstrapping Language-Image Pretraining) model to automatically generate captions and detailed descriptions for uploaded images. Built with Flask, PyTorch, and the Hugging Face Transformers library, it provides an intuitive interface for users to upload images and receive AI-generated descriptions.

### System Architecture

#### Backend Components
1. **Flask Application (`app.py`)**
   - Serves as the main application server
   - Handles routing and file uploads
   - Integrates with the BLIP model for image processing
   - Key routes:
     - `GET /`: Serves the main upload interface
     - `POST /upload`: Processes image uploads and generates captions
     - `GET /uploads/<filename>`: Serves uploaded images

2. **BLIP Model Integration**
   - Uses `BlipProcessor` for image preprocessing
   - Implements `BlipForConditionalGeneration` for caption generation
   - Model initialization:
   ```python
   processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
   model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
   ```

#### Frontend Components
1. **Main Interface (`index.html`)**
   - Responsive design using flexbox
   - File upload form with immediate feedback
   - Dynamic result display section
   - Styled with embedded CSS for simplified deployment

2. **JavaScript Functionality**
   - Asynchronous form submission
   - Dynamic content updates without page reload
   - Error handling and user feedback

### Installation and Setup

#### Prerequisites
- Python 3.7 or higher
- pip package manager
- Git (for cloning the repository)

#### Dependencies
```
Flask==2.3.2
torch==2.1.0
torchvision==0.15.2
transformers==4.30.0
Pillow==9.5.0
```

#### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Ganesh2409/ImageCaptioningAndDescription.git
   cd ImageCaptioningAndDescription
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Project Structure
```
ImageCaptioningAndDescription/
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
├── uploads/              # Directory for uploaded images
├── templates/
│   └── index.html        # Main web interface
└── static/
    └── css/
        └── styles.css    # CSS styles (if separated)
```

### API Documentation

#### 1. Main Upload Interface
- **Endpoint**: `GET /`
- **Purpose**: Serves the main application interface
- **Response**: HTML page with upload form
- **Example Usage**: Navigate to `http://localhost:5000/`

#### 2. Image Upload and Processing
- **Endpoint**: `POST /upload`
- **Purpose**: Process uploaded images and generate captions
- **Request Format**: Multipart form data with 'file' field
- **Response Format**:
  ```json
  {
      "caption": "Generated image caption",
      "detailed_description": "Detailed image description",
      "image_url": "/uploads/filename.jpg"
  }
  ```
- **Example Usage**:
  ```bash
  curl -X POST -F "file=@image.jpg" http://localhost:5000/upload
  ```

#### 3. Image Serving
- **Endpoint**: `GET /uploads/<filename>`
- **Purpose**: Serve uploaded images
- **Response**: Image file
- **Example**: `http://localhost:5000/uploads/image.jpg`


#### Frontend Implementation
1. **Form Submission**
   ```javascript
   const formData = new FormData(form);
   const response = await fetch('/upload', {
       method: 'POST',
       body: formData
   });
   ```

2. **Result Display**
   ```javascript
   document.getElementById('caption').textContent = data.caption;
   document.getElementById('detailed-description').textContent = data.detailed_description;
   document.getElementById('uploaded-image').src = data.image_url;
   ```

### Security Considerations
1. **File Upload Security**
   - Limited to image files only
   - File size should be restricted (implement in production)
   - Secure filename handling recommended

2. **Server Security**
   - Debug mode should be disabled in production
   - Implement proper error handling
   - Consider adding authentication for production use

### Development and Testing

#### Local Development
1. Start the development server:
   ```bash
   python app.py
   ```
2. Access the application at `http://localhost:5000`
3. Use debug mode for development purposes

#### Testing
- Implement unit tests for the Flask routes
- Test image processing pipeline
- Verify error handling
- Test different image formats and sizes

### Deployment Considerations

#### Production Setup
1. Use a production-grade WSGI server (e.g., Gunicorn)
2. Implement proper logging
3. Set up error monitoring
4. Configure appropriate security headers

#### Server Configuration
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Troubleshooting Guide

#### Common Issues
1. **Model Loading Errors**
   - Verify PyTorch installation
   - Check CUDA compatibility if using GPU
   - Ensure sufficient system memory

2. **Image Upload Issues**
   - Check file permissions in uploads directory
   - Verify file size limits
   - Ensure proper form encoding

3. **Caption Generation Issues**
   - Verify model initialization
   - Check input image format
   - Monitor memory usage

### Future Enhancements
1. **Technical Improvements**
   - Implement image compression
   - Add support for batch processing
   - Optimize model loading time
   - Add caching mechanisms

2. **Feature Additions**
   - Multiple model support
   - Custom model fine-tuning
   - User authentication
   - Image history tracking

3. **UI Enhancements**
   - Progress indicators
   - Drag-and-drop upload
   - Preview functionality
   - Response time optimization


# Shipment Delay Prediction Web Application Project-2 
## Technical Documentation

### Overview
This web application utilizes machine learning to predict shipment delays based on various input parameters such as weather conditions, traffic conditions, shipment details, and planned delivery dates. Built with Flask and scikit-learn, it provides a user-friendly interface for logistics professionals to assess the likelihood of shipment delays.

### System Architecture

#### Backend Components
1. **Flask Application (`app.py`)**
   - Serves as the main application server
   - Handles form submissions and predictions
   - Manages model loading and inference
   - Key routes:
     - `GET /`: Serves the main prediction interface
     - `POST /`: Processes form submissions and returns predictions

2. **Machine Learning Components**
   - Pre-trained model (`model_log.pkl`)
   - Column transformer (`column_transformer.pkl`)
   - Handles feature preprocessing and prediction logic

#### Frontend Components
1. **Main Interface (`index.html`)**
   - Input form for shipment details
   - Real-time prediction display
   - Responsive design for various screen sizes
   - User-friendly error messages

### Installation and Setup

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

#### Dependencies
```
Flask==2.3.2
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
pickle-mixin==1.0.2
```

#### Installation Steps
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd shipment-delay-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Project Structure
```
shipment-delay-prediction/
├── app.py                     # Main Flask application
├── requirements.txt           # Project dependencies
├── model_files/              # ML model files
│   ├── model_log.pkl         # Trained prediction model
│   └── column_transformer.pkl # Feature transformer
├── static/                   # Static assets
│   └── css/                  # CSS stylesheets
├── templates/                # HTML templates
│   └── index.html           # Main prediction interface
├── data/                    # Training data directory
├── Delay_Prediction.ipynb   # Model training notebook
├── Delay_Prediction.py      # Training script
└── README.md                # Project documentation
```

### Feature Details

#### Input Parameters
1. **Shipment Information**
   - Shipment ID (tracking purposes only)
   - Origin location
   - Destination location
   - Shipment date
   - Planned delivery date
   - Vehicle type
   - Distance (km)

2. **Environmental Factors**
   - Weather conditions
   - Traffic conditions

#### Derived Features
1. **Planned Shipment Gap**
   - Calculated as the difference between planned delivery date and shipment date
   - Used as a key predictor in the model

### Implementation Details

#### Data Processing Pipeline

1. **Feature Transformation**
   ```python
   # Load column transformer
   column_transformer = pickle.load(open('model_files/column_transformer.pkl', 'rb'))
   
   # Transform features
   X_transformed = column_transformer.transform(input_data)
   ```

2. **Prediction Generation**
   ```python
   # Load model and make prediction
   model = pickle.load(open('model_files/model_log.pkl', 'rb'))
   prediction = model.predict(X_transformed)
   ```

### Machine Learning Model

#### Model Details
- Algorithm: Logistic Regression (or specified algorithm)
- Input features: 
  - Categorical: Origin, Destination, Vehicle Type, Weather, Traffic
  - Numerical: Distance, Planned Shipment Gap
- Output: Binary classification (On Time/Delayed)

#### Model Training Process
1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features

2. **Feature Engineering**
   - Create Planned Shipment Gap
   - Process categorical variables
   - Handle date-related features

3. **Model Training**
   - Split data into training and testing sets
   - Train model using preprocessed data
   - Evaluate model performance
   - Save model and transformer

### API Documentation

#### Prediction Endpoint
- **Route**: `POST /`
- **Purpose**: Process shipment details and return delay prediction
- **Input Format**: Form data with shipment details
- **Response Format**: Prediction result ("On Time" or "Delayed")
- **Example Usage**:
  ```python
  response = requests.post('http://localhost:5000/', data={
      'shipment_id': 'SHP001',
      'origin': 'Chennai',
      'destination': 'Mumbai',
      'shipment_date': '2024-01-01',
      'planned_delivery_date': '2024-01-05',
      'vehicle_type': 'Truck',
      'distance': '4500',
      'weather_conditions': 'Clear',
      'traffic_conditions': 'Moderate'
  })
  ```

### Error Handling

#### Common Errors and Solutions
1. **Invalid Date Format**
   - Error: "Invalid date format"
   - Solution: Ensure dates are in YYYY-MM-DD format

2. **Missing Fields**
   - Error: "All fields are required"
   - Solution: Complete all form fields

3. **Invalid Distance**
   - Error: "Distance must be a positive number"
   - Solution: Enter valid numerical distance

### Deployment Guidelines

#### Local Deployment
1. Run Flask development server:
   ```bash
   python app.py
   ```
2. Access application at `http://localhost:5000`

### Testing Guidelines

#### Unit Testing
1. Test input validation
2. Test prediction functionality
3. Test error handling
4. Test date calculations

#### Integration Testing
1. Test end-to-end prediction flow
2. Test form submission
3. Test model loading
4. Test feature transformation

### Maintenance and Updates

#### Regular Maintenance Tasks
1. Update dependencies
2. Monitor model performance
3. Retrain model with new data
4. Update documentation

#### Model Updates
1. Collect new training data
2. Evaluate model performance
3. Retrain if necessary
4. Update model files

