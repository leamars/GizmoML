import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from threading import Thread
import imutils
import dlib
import cv2
import playsound

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def sound_alarm(path):
	playsound.playsound(path, False)
	
thresh = 0.40
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0

soundFilePath = "lullaby.wav"
alarm_on= False

img_path = 'overlay.jpg'
img = cv2.imread(img_path, -1)

screenWidth = 800
screenHeightMult = 3/5

overlayOn = False

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=screenWidth)
	overlay = frame.copy()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:

				# Green filled in rectangle
				frame_shape = frame.shape
				print(frame_shape)
				print(len(frame[:0]))
				
				cv2.rectangle(overlay,(0, 0),(800, 450),(0,140,222),-1)
				opacity = 0.4
				cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
				cv2.putText(frame, "****************SWEET DREAMS ;)****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
				cv2.putText(frame, "****************SWEET DREAMS ;)****************", (10,400),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
				
				if not alarm_on:
					alarm_on = True
					sound_alarm(soundFilePath)
		else:
			flag = 0
			alarm_on = False

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()
