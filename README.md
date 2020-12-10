# The Handler: WebCam based Hand Gesture Control Tool

## Summary

## Technologies Used
Programming Language: Python\
Libraries Used: PyTorch, OpenCV, PyAutoGUI\

## Summary
This project is a tool that recognizes hand gestures via a live WebCam and performs mouse functions accordingly. The motivation behind this project is to replace the traditional and cumbersome way of interacting with computers with gesture based human interaction. This system is a way of interacting with computer using a real time recognition of dynamic hand gestures from video streams through webcam. The proposed solution uses 3D-CNN models for the live-video (gesture) classification purposes due to its superior ability of extracting spatio-temporal features within video frames. And based on this gesture detection, activities like scrolling, switching slides, zooming in and out, etc are automated using powerful python libraries. Additionally we have also used transfer learning with MobileNetV2 model to provide each user an option to customise their own gestures for each class of actions. We were able to achieve more than 90% accuracy in recognizing the custom gestures set by user in the real-time based system with our solution. 

The project is divided into two parts - Server and Client (codes can be found in src folder). \
At First, the Client runs the `input_utils.py` that collects the data of user. Then, the file `client.py` needs to be executed which zips the data collected by the user and make a HTTP POST request (REST API call) to the server and sends the zipped data. Then, the Server sends a trained model (which was trained on the user data) in the response of the HTTP POST request. Then at the client side, the inference is performed using `controls.py` file that classifies the live hand gesture using two files - `yolo.py` and `main.py` and with the help of PyAutoGUI library performs that particular function. The `yolo.py` file helps to perform the finger tracking which accurately performs scrolling functions. And the `main.py` file uses the MobileNetV2 model that came as a response from the server to classify the hand gesture in the pre defined classes. \
The Server side code is responsible for receiving the data from client, training the model on the received data and sending the model to the client. The server uses Apache2 HTTP server, Gunicorn WSGI server, and Flask micro-framework. The model is written using PyTorch library. The file `app.py` receives requests from clients and calls the model training function which is mentioned in `main.py`. It trains the model on the received data and then model is returned to the client. \
\




