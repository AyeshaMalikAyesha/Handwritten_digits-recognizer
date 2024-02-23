from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf

# Initialize Flask application
app = Flask(__name__)

# Load pre-trained TensorFlow model
model = tf.keras.models.load_model('custom_handwritten_digits_model.model')


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image_data):
    try:
        # Decode base64 image data
        image_data_decoded = base64.b64decode(image_data.split(',')[1])
        # Open image from decoded bytes
        image = Image.open(BytesIO(image_data_decoded))
        # Convert image to grayscale
        image = image.convert('L')
        # Resize image to 28x28 pixels
        image = image.resize((28, 28))
        # Convert image to numpy array
        image_array = np.array(image)
        # Normalize pixel values
        image_array = image_array / 255.0
        # Reshape image array for model input
        image_array = np.reshape(image_array, (1, 28, 28, 1))
        return image_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/recognize_digit', methods=['POST'])
def recognize_digit():
    try:
        # Get drawing from canvas
        image_data = request.form.get('imageData')
        
        if image_data is None:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image data
        image = preprocess_image(image_data)
        
        # Predict digit using the model
        prediction = model.predict(image)
        
        # Get predicted digit and probabilities
        digit = np.argmax(prediction)
        probabilities = prediction[0]
        
        # Convert numpy int64 to int for JSON serialization
        digit = int(digit)
        
        # Return JSON response
        return jsonify({'digit': digit, 'probabilities': probabilities.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

