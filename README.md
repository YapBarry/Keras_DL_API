# Keras_DL_API
## This repository loads a pre-trained ResNet50 model and uses it to predict the class of objects in the pictures you feed it with.
## Python version 3.6.7


Run the following code in the root folder:
```
pip install requirements.txt
```
```
python predict_app.py
```
Above code loads the model (ResNet50) and runs the flask app.

**Please wait for the model to be fully loaded before you run the next line of code. You should only run it after seeing the below output in your terminal**
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

1) For testing of API (/predict) endpoint via python script:

Uncomment line 56 in predict_app.py. <br>
Comment out lines 59-61 in predict_app.py.
```
python test_app.py
```
Above code will POST a request (consisting of a jpg file - tennis_ball2.jpg) to the server. <br>
The server would then preprocess the data and then pass it to the model.<br>
The prediction of the model will then be shown.
```
(BarryEnv) Users-MacBook-Pro:Keras_DL_API user$ python test_app.py
tennis_ball: 0.9986
```

2) For uploading of image locally using frontend:

Uncomment lines 59-61 in predict_app.py. <br>
Comment out line 56 in predict_app.py.

On your web browser, navigate to the following address: 
http://127.0.0.1:5000/static/predict.html

<img width="400" alt="Screenshot 2021-09-26 at 10 24 17 PM" src="https://user-images.githubusercontent.com/58761788/134811927-7c24adc2-10d0-4026-9609-9a2552f1b8bf.png">

Click on "choose file" button to select and upload your desired picture.
Once done, click on the "predict" button to send your file to the server for prediction.
You should see something similar to the screenshot posted below:

<img width="694" alt="Screenshot 2021-09-26 at 10 29 25 PM" src="https://user-images.githubusercontent.com/58761788/134812130-d304fbd7-7b29-420b-ad89-62de90329e11.png">
