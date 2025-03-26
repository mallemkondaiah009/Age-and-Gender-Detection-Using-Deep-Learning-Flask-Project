import cv2
import numpy as np
import argparse
from flask import Flask, render_template, Response, request
from PIL import Image
import io
import webbrowser
import logging

UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def highlightFace(net, frame, conf_threshold=0.7):
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
            x1 = int(detections[0, 0, i,3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def gen_frames():
    faceProto = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\opencv_face_detector_uint8.pb"
    faceModel = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\opencv_face_detector.pbtxt"
    ageProto = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\age_deploy.prototxt"
    ageModel = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\age_net.caffemodel"
    genderProto = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\gender_deploy.prototxt"
    genderModel = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(0)
    padding = 20
    web_link_opened = False

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
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
            age_index = agePreds[0].argmax()
            age = ageList[age_index]

            if age_index == 4 and not web_link_opened:
                webbrowser.open('https://docs.google.com/presentation/d/1K1o8Y8KV3hrWV0lzDBsQkTn_6Xm945U8/edit?usp=drive_link&ouid=115243913653739313402&rtpof=true&sd=true')
                web_link_opened = True

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        ret, encodedImg = cv2.imencode('.jpg', resultImg)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')


def gen_frames_photo(img_array):
    faceProto = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\opencv_face_detector_uint8.pb"
    faceModel = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\opencv_face_detector.pbtxt"
    ageProto = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\age_deploy.prototxt"
    ageModel = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\age_net.caffemodel"
    genderProto = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\gender_deploy.prototxt"
    genderModel = "C:\\Users\\Mallem Kondaiah\\OneDrive\\Desktop\\projects\\Age-and-Gender-Detection-Using-Deep-Learning-Flask-Project\\Age and Gender Detection Using Deep Learning Flask Project\\gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    try:
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error in cvtColor: {e}")
        return f"Error in cvtColor: {e}"

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

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    ret, encodedImg = cv2.imencode('.jpg', resultImg)
    if not ret:
        logging.error("Error encoding the image")
        return "Error encoding the image"

    logging.info("Frame processed and encoded successfully")
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')

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
        file = request.files['fileToUpload']
        if file:
            try:
                img = Image.open(io.BytesIO(file.read()))
                img_array = np.array(img)
                if img_array.shape[2] == 4:  # If RGBA, convert to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                logging.info("Image uploaded and processed successfully")
                return Response(gen_frames_photo(img_array), mimetype='multipart/x-mixed-replace; boundary=frame')
            except Exception as e:
                logging.error(f"Error in processing the image: {e}")
                return f"Error in processing the image: {e}"
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)