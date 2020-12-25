import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_PLAIN


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:

        landmarks = predictor(gray, face)

        # le stands for left eye
        # le coordinates and lines
        le_left_point = (landmarks.part(36).x, landmarks.part(36).y)
        le_right_point = (landmarks.part(39).x, landmarks.part(39).y)
        le_center_top = midpoint(landmarks.part(37), landmarks.part(38))
        le_center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        # re stands for right eye
        # re coordinates and lines
        re_left_point = (landmarks.part(42).x, landmarks.part(42).y)
        re_right_point = (landmarks.part(45).x, landmarks.part(45).y)
        re_center_top = midpoint(landmarks.part(43), landmarks.part(44))
        re_center_bottom = midpoint(landmarks.part(47), landmarks.part(46))

        # left eye vertical line length
        le_ver_length = hypot(
            (le_center_top[0] - le_center_bottom[0]), (le_center_top[1] - le_center_bottom[1]))

        # left eye horizontal line length
        le_hor_length = hypot(
            (le_right_point[0] - le_left_point[0]), (le_right_point[1] - le_left_point[1]))

        # right eye vertical line length
        re_ver_length = hypot(
            (re_center_top[0] - re_center_bottom[0]), (re_center_top[1] - re_center_bottom[1]))

        # right eye horizontal line length
        re_hor_length = hypot(
            (re_right_point[0] - re_left_point[0]), (re_right_point[1] - re_left_point[1]))

        ratio = (le_hor_length + re_hor_length) / \
            (le_ver_length + re_ver_length)

        if ratio > 5.6:
            cv2.putText(frame, "Blinking", (50, 150), font, 3, (255, 0, 0))

        # gaze detection
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye_region], True, (0, 0, 255), 2)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[0, :])
        max_y = np.max(left_eye_region[0, :])

        #eye = frame[min_y:max_y, min_x:max_x]
        #eye = cv2.resize(eye, None, fx=5, fy=5)
        #cv2.imshow("Eye", eye)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
