# Keras_DL_API

Run the following code in the root folder:

```
python predict_app.py
```
Above code loads the model (ResNet50) and runs the flask app.

**Please wait for the model to be fully loaded before you run the next line of code. You should only run it after seeing the below output in your terminal**
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

```
python test_app.py
```
Above code will POST a request (consisting of a jpg file - tennis_ball2.jpg) to the server. <br>
The server would then preprocess the data and then pass it to the model.<br>
The prediction of the model will then be shown.
```
(BarryEnv) Users-MacBook-Pro:Assessment user$ python test_app.py
1. tennis_ball: 0.9986
2. racket: 0.0013
3. ping-pong_ball: 0.0000
4. soccer_ball: 0.0000
5. safety_pin: 0.0000
```
