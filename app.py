# Step 2: Your imports and setup
from flask import Flask, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Step 3: Configurations
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model (make sure the file is uploaded to /content/)
# Note: Ensure your model file "/content/DiseaseDetection.h5" exists.
# If not, you will need to save the trained model first.
try:
    model = load_model("DiseaseDetection.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error, e.g., exit or load a dummy model
    # For demonstration, we'll print the error and continue, but in a real app,
    # you'd want to handle this more robustly.

# Define your label map based on the unique labels in your training data
# This needs to match the Labels_map created when preparing the training data.
# You defined it earlier as:
# Labels_map = {folder: idx for idx, folder in enumerate(Train_subfolders)}
# You should use the same mapping here. For example
Labels_map = {'Darier_s disease': 0, 'Muehrck-e_s lines': 1, 'aloperia areata': 2, 'beau_s lines': 3, 'bluish nail': 4, 'clubbing': 5, 'eczema': 6, 'half and half nailes (Lindsay_s nails)': 7, 'koilonychia': 8, 'leukonychia': 9, 'onycholycis': 10, 'pale nail': 11, 'red lunula': 12, 'splinter hemmorrage': 13, 'terry_s nail': 14, 'white nail': 15, 'yellow nails': 16}
reverse_labels_map = {v: k for k, v in Labels_map.items()}
num_classes = len(reverse_labels_map)

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Step 4: Define routes
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Disease Detection Upload</title>
  <style>
    body {
      background: linear-gradient(to right, #2c3e50, #3498db);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: #fff;
      text-align: center;
      padding: 50px;
    }
    .container {
      background-color: rgba(255, 255, 255, 0.1);
      border-radius: 15px;
      padding: 40px;
      max-width: 500px;
      margin: auto;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    h1 {
      font-size: 2.5em;
      margin-bottom: 20px;
      color: #f1c40f;
    }
    input[type="file"] {
      display: block;
      margin: 20px auto;
      padding: 10px;
      font-size: 1.1em;
      border-radius: 10px;
      background-color: #fff;
      color: #000;
      cursor: pointer;
      border: none;
    }
    input[type="submit"] {
      background-color: #27ae60;
      color: white;
      padding: 12px 25px;
      font-size: 1.1em;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      transition: 0.3s;
    }
    input[type="submit"]:hover {
      background-color: #2ecc71;
      transform: scale(1.05);
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🌿 Upload an Image to Detect Disease 🌿</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file" required>
      <input type="submit" value="Upload & Predict">
    </form>
  </div>
</body>
</html>
'''


@app.route('/', methods=['POST'])
def upload_and_predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img = Image.open(filepath).resize((224, 224)).convert('RGB')
            x = np.array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)

            # Ensure the model was loaded successfully before predicting
            if 'model' in globals() and model is not None:
                preds = model.predict(x)
                print(preds)
                predicted_class_index = np.argmax(preds[0])
                predicted_label = reverse_labels_map.get(predicted_class_index, "Unknown")
                message = f"The Detected Condition is: {predicted_label}"
            else:
                message = "Error: Model not loaded."

        except Exception as e:
            message = f"Error during prediction: {str(e)}"
        finally:
            # Ensure filepath exists before trying to remove it
            if os.path.exists(filepath):
                os.remove(filepath)

            return f'''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Prediction Result</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #1abc9c, #2c3e50);
      color: #fff;
      text-align: center;
      height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    .result-box {{
      background: rgba(255, 255, 255, 0.1);
      border-radius: 15px;
      padding: 50px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
      max-width: 600px;
    }}
    h1 {{
      font-size: 2.8em;
      margin-bottom: 20px;
      color: #f1c40f;
      text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }}
    p {{
      font-size: 1.5em;
      margin-bottom: 30px;
    }}
    a {{
      display: inline-block;
      padding: 12px 25px;
      font-size: 1.2em;
      background-color: #e74c3c;
      color: white;
      text-decoration: none;
      border-radius: 10px;
      transition: background-color 0.3s, transform 0.2s;
    }}
    a:hover {{
      background-color: #c0392b;
      transform: scale(1.05);
    }}
  </style>
</head>
<body>
  <div class="result-box">
    <h1>🌟 Prediction Result 🌟</h1>
    <p>{message}</p>
    <a href="/">🔁 Upload Another Image</a>
  </div>
</body>
</html>
'''

    return 'Invalid file or file type'

# Step 5: Launch app using pyngrok
# Replace this string with your actual token
# authtoken = "YOUR_NGROK_AUTH_TOKEN" # <-- Replace with your token
# conf.get_default().auth_token = authtoken

# It seems you've already set the token in a previous cell,
# so you might not need this line again if you run the cells sequentially.

# Ensure you've set your ngrok auth token before this step.
# You can do this by running !ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
# or programmatically as you did in the previous cell.
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
