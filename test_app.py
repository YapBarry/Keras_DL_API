# ALTERNATE WAYS TO TEST AP USING CURL
# curl -X POST -F image=@your_picture.jpeg "http://localhost:5000/predict"

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "tennis_ball2.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read() #open(file_path,"rb") basically prepares the file to be read in b(bytes) form and then adding.read() means pulling the data from the file
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# loop over the predictions and display them
for (i, result) in enumerate(r["predictions"]):
    print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))
