from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
from docsapp.obj_inference import inference

app = Flask(__name__)

# Load the YOLO model globally when the app starts
model = YOLO('./model/best.pt')

@app.route('/')
def index():
    return "YOLO Inference API is Running!"

# Route for image inference
@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Get the image path from the request (in real case, might come from a form or API payload)
        image_path = request.json.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Invalid image path"}), 400

        # Perform inference
        result = inference(model, image_path)
        
        # Return the result as JSON
        return jsonify({"status": "success", "cropped_images_count": len(result['result'])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
