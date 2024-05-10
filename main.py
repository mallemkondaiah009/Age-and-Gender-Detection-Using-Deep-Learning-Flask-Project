import cv2
import numpy as np
import webbrowser

def gen_frames():
    faceProto = "Age and Gender Detection Using Deep Learning Flask Project/opencv_face_detector_uint8.pb"
    faceModel = "Age and Gender Detection Using Deep Learning Flask Project/opencv_face_detector.pbtxt"
    ageProto = "Age and Gender Detection Using Deep Learning Flask Project/age_deploy.prototxt"
    ageModel = "Age and Gender Detection Using Deep Learning Flask Project/age_net.caffemodel"
    genderProto = "Age and Gender Detection Using Deep Learning Flask Project/gender_deploy.prototxt"
    genderModel = "Age and Gender Detection Using Deep Learning Flask Project/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(0)
    padding = 20

    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        frame = cv2.flip(frame, 1)
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")
            continue

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age_range = ageList[agePreds[0].argmax()]
            print(f'Age Range: {age_range}')

            cv2.putText(resultImg, f'{gender}, {age_range}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if age_range == '(25-32)' or age_range == '(38-43)' or age_range == '(48-53)' or age_range == '(60-100)':
                webbrowser.open("https://www.hackerrank.com")  # Open HackerRank website

        if resultImg is None:
            continue

        ret, encodedImg = cv2.imencode('.jpg', resultImg)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImg) + b'\r\n')

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# Main function
if __name__ == "__main__":
    for frame in gen_frames():
        cv2.imshow("Detecting age and gender", cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), -1))
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()