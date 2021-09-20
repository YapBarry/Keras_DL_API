# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "tennis_ball2.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# loop over the predictions and display them
# for (i, result) in enumerate(r["predictions"]):
#     print("{}. {}: {:.4f}".format(
#         i + 1, result["label"], result["probability"]))
print("{}: {:.4f}".format(r["predictions"]
      ["label"], r["predictions"]["probability"]))
