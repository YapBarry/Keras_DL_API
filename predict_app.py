# Import the necessary packages
import numpy as np
from numpy import asarray
import time

from PIL import Image
from io import BytesIO

from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from keras.applications import imagenet_utils

from flask import Flask, jsonify, request

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None

def load_model():
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        rgb = image.convert('RGB')
    else:
        rgb = image
    
    # Resize image to have a width and height of 64
    rgb = rgb.resize((224,224))
    
    # Convert the image to a numpy array
    pixels = asarray(rgb)
    
    # Expand dimensions of image to include batch size as model was trained with inclusion of batch size dimension
    pixels = np.expand_dims(pixels, 0)
    
    # Preprocess Image's array values to be within -1 to 1
    pixels = imagenet_utils.preprocess_input(pixels)
    
    return pixels

@app.route("/predict", methods=["POST"])
def predict():
    # measuring start time
    start = time.perf_counter()
    
    # read the image in PIL format
    # file storage class without.read(), with.read() is bytes class/object
    image = request.files["image"].read() 
    
    # BytesIO object keeps the image(a byte class) in an in-memory buffer (chunk of ram) and then we call the PIL Image.open method on it
    # PIL's Image.open only accepts file like object and bytes object is not file object, so we need to keep it in 
    # a io.BytesIO object which is a file object and then only we can call it using PIL's Image.open    
    image = Image.open(BytesIO(image)) 
    
    # preprocess image
    image = prepare_image(image)

    # predict result
    preds = model.predict(image)
    results = imagenet_utils.decode_predictions(preds) # returns list of lists containing (class_name, class_description, score)
    data = []

    # loop over the results and add them to the list of
    # returned predictions
    for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": float(prob)}
        data.append(r)
    
    # measuring ending time
    end = time.perf_counter()
    time_diff = end - start
    
    response = {
        'predictions': data,
        'response_time': time_diff
    }

    
    return jsonify(response)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()



