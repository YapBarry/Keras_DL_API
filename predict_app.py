# import the necessary packages
import base64
import numpy as np
from numpy import asarray
import time

from PIL import Image
from io import BytesIO

from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from keras.applications import imagenet_utils

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# initialize our Flask application and the Keras model
app = Flask(__name__)
CORS(app)
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

    # resize image to have a width and height of 64
    rgb = rgb.resize((224, 224))

    # convert the image to a numpy array
    pixels = asarray(rgb)

    # expand dimensions of image to include batch size as model was trained with inclusion of batch size dimension
    pixels = np.expand_dims(pixels, 0)

    # preprocess Image's array values to be within -1 to 1
    pixels = imagenet_utils.preprocess_input(pixels)

    return pixels


@app.route("/predict", methods=["POST"])
def predict():
    # measuring start time
    start = time.perf_counter()

    # uncomment below code if you are only testing api endpoint using test_app.py script
    # image = request.files["image"].read()

    # uncomment below code when we are testing frontend of the app via http://127.0.0.1:5000/static/predict.html
    message = request.get_json(force=True)
    encoded_image = message['image']
    image = base64.b64decode(encoded_image)

    image = Image.open(BytesIO(image))

    # preprocess image
    image = prepare_image(image)

    # predict result
    preds = model.predict(image)
    # returns list of lists containing (class_name, class_description, score)
    results = imagenet_utils.decode_predictions(preds)
    data = []

    # loop over the results and add them to the list of
    # returned predictions
    for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": float(prob)}
        data.append(r)

    # measuring ending time
    end = time.perf_counter()
    time_diff = end - start

    # change to return only top result in data instead of top 5
    data = data[0]

    response = {
        'predictions': data,
        'response_time': time_diff
    }
    return jsonify(response)


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(debug=True)
