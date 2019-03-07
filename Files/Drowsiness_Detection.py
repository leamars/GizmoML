import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from threading import Thread
import imutils
import dlib
import cv2
import playsound
import pygame

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def sound_alarm(on):
	if on:
		pygame.mixer.music.play()
	else:
		pygame.mixer.music.stop()
	
# Initialize pygame music player 
soundFilePath = "Gentle-wake-alarm-clock.mp3"
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(soundFilePath)

thresh = 0.35
frame_check = 40
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap=cv2.VideoCapture(0)
flag=0
alarm_on= False
screen_width = 800

font = cv2.FONT_HERSHEY_DUPLEX
text = "How about a movie?"
text_size = cv2.getTextSize(text, font, 1, 2)
text_width = text_size[0][0]
text_height = text_size[0][1]

catch_me_path = "catchMe.jpg"
catch_img = cv2.imread(catch_me_path, -1)

wizard_path = "wizard.jpg"
wizard_img = cv2.imread(wizard_path, -1)

forest_path = "forrest.jpg"
forest_img = cv2.imread(forest_path, -1)

show_screen_message = False

# Frame shape is basically the size of the frame, so that we can make our overlay the same size, and
# center the text in the middle.

def showBlueOverlay(frame_shape):
	cv2.rectangle(overlay,(0, 0),(frame_shape[1], frame_shape[0]),(203, 139, 48),-1)
	opacity = 0.3
	cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

def showOnScreenMessage(frame_shape, should_show):
	if should_show:
		middle_x = frame_shape[1]/2
		middle_y = frame_shape[0]/2

		padding = 20

		text_x = int(middle_x - text_width/2)
		text_y = int(middle_y - text_height/2 - padding)
		cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

		catch_img_x = int(middle_x - catch_img.shape[1]/2)
		catch_img_y = int(middle_y - catch_img.shape[0]/2 + text_height + padding * 2)

		frame[catch_img_y: catch_img_y + catch_img.shape[0], catch_img_x: catch_img_x + catch_img.shape[1]] = catch_img

		forest_img_x = int(catch_img_x - catch_img.shape[1] - padding)
		forest_img_y = catch_img_y

		frame[forest_img_y: forest_img_y + forest_img.shape[0], forest_img_x: forest_img_x + forest_img.shape[1]] = forest_img

		wizard_img_x = int(catch_img_x + catch_img.shape[1] + padding)
		wizard_img_y = catch_img_y

		frame[wizard_img_y: wizard_img_y + wizard_img.shape[0], wizard_img_x: wizard_img_x + wizard_img.shape[1]] = wizard_img

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=screen_width)
	frame_shape = frame.shape

	# First create the image with alpha channel
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	# Then assign the mask to the last channel of the image
	#frame[:, :, 3] = 1
	
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
		cv2.drawContours(frame, [leftEyeHull], -1, (203, 139, 48), 2)
		cv2.drawContours(frame, [rightEyeHull], -1, (203, 139, 48), 2)

		if ear < thresh:
			flag += 1
			#print (flag)
			if flag >= frame_check:
				show_screen_message = False
				showBlueOverlay(frame_shape)

				if not alarm_on:
					alarm_on = True
					sound_alarm(alarm_on)
		else:
			flag = 0
			if alarm_on:
				show_screen_message = True
			alarm_on = False
			sound_alarm(alarm_on)

		showOnScreenMessage(frame_shape, show_screen_message)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	if key == ord("r"):
		# Reset to nothing showing
		show_screen_message = False

cv2.destroyAllWindows()
cap.stop()
