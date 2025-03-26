import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, request
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Flask app and upload folder
UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to highlight faces in the frame
def highlightFace(net, frame, conf_threshold=0.7):
    try:
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, faceBoxes
    except Exception as e:
        logging.error(f"Error in highlightFace: {e}")
        return frame, []

# Function to generate frames from the webcam
def gen_frames():
    # Model paths (using relative paths)
    MODEL_PATHS = {
        'face_proto': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\opencv_face_detector_uint8.pb',  # Binary weights file
        'face_model': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\opencv_face_detector.pbtxt',    # Text config file
        'age_proto': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\age_deploy.prototxt',            # Text config file
        'age_model': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\age_net.caffemodel',             # Binary weights file
        'gender_proto': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\gender_deploy.prototxt',      # Text config file
        'gender_model': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\gender_net.caffemodel'        # Binary weights file
    }

    # Verify all model files exist
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            logging.error(f"Model file missing: {path}")
            return

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    try:
        # Correct argument order: weights file first, config file second
        faceNet = cv2.dnn.readNet(MODEL_PATHS['face_proto'], MODEL_PATHS['face_model'])
        ageNet = cv2.dnn.readNet(MODEL_PATHS['age_model'], MODEL_PATHS['age_proto'])
        genderNet = cv2.dnn.readNet(MODEL_PATHS['gender_model'], MODEL_PATHS['gender_proto'])
    except cv2.error as e:
        logging.error(f"Failed to load models: {e}")
        return

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        logging.error("Failed to open webcam")
        return

    padding = 20
    try:
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                logging.warning("No frame captured from webcam")
                break

            frame = cv2.flip(frame, 1)
            resultImg, faceBoxes = highlightFace(faceNet, frame)

            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                            max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]

                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            ret, encodedImg = cv2.imencode('.jpg', resultImg)
            if not ret:
                logging.error("Failed to encode frame")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')
    finally:
        video.release()

# Function to process uploaded images
def gen_frames_photo(img_array):
    # Model paths (using relative paths)
    MODEL_PATHS = {
        'face_proto': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\opencv_face_detector_uint8.pb',  # Binary weights file
        'face_model': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\opencv_face_detector.pbtxt',    # Text config file
        'age_proto': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\age_deploy.prototxt',            # Text config file
        'age_model': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\age_net.caffemodel',             # Binary weights file
        'gender_proto': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\gender_deploy.prototxt',      # Text config file
        'gender_model': 'C:\\Users\\Mallem Kondaiah\\OneDrive\\Documents\\MyFiles\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\models\\gender_net.caffemodel'        # Binary weights file
    }

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            logging.error(f"Model file missing: {path}")
            return b''

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    try:
        # Correct argument order: weights file first, config file second
        faceNet = cv2.dnn.readNet(MODEL_PATHS['face_proto'], MODEL_PATHS['face_model'])
        ageNet = cv2.dnn.readNet(MODEL_PATHS['age_model'], MODEL_PATHS['age_proto'])
        genderNet = cv2.dnn.readNet(MODEL_PATHS['gender_model'], MODEL_PATHS['gender_proto'])
    except cv2.error as e:
        logging.error(f"Failed to load models: {e}")
        return b''

    try:
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error in cvtColor: {e}")
        return f"Error in cvtColor: {e}".encode()

    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                    max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    ret, encodedImg = cv2.imencode('.jpg', resultImg)
    if not ret:
        logging.error("Error encoding the image")
        return b"Error encoding the image"

    logging.info("Frame processed and encoded successfully")
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'fileToUpload' not in request.files:
            return "No file part in the request", 400
        file = request.files['fileToUpload']
        if file.filename == '':
            return "No selected file", 400
        try:
            img = Image.open(io.BytesIO(file.read()))
            img_array = np.array(img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # If RGBA, convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            logging.info("Image uploaded and processed successfully")
            return Response(gen_frames_photo(img_array), mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            logging.error(f"Error in processing the image: {e}")
            return f"Error in processing the image: {e}", 500
    return render_template('photo.html')

if __name__ == '__main__':
    app.run(debug=True)