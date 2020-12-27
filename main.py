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


def getBlinkingRatio(facial_lanmarks, points):

    left_point = (facial_lanmarks.part(
        points[0]).x, facial_lanmarks.part(points[0]).y)
    right_point = (facial_lanmarks.part(
        points[1]).x, facial_lanmarks.part(points[1]).y)
    center_top = midpoint(facial_lanmarks.part(
        points[2]), facial_lanmarks.part(points[3]))
    center_bottom = midpoint(facial_lanmarks.part(
        points[4]), facial_lanmarks.part(points[5]))

    ver_length = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    hor_length = hypot(
        (right_point[0] - left_point[0]), (right_point[1] - left_point[1]))

    return hor_length, ver_length


def getGazeRatio(facial_lanmarks, points, gray):

    eye_region = np.array([(facial_lanmarks.part(points[0]).x,
                            facial_lanmarks.part(points[0]).y),
                           (facial_lanmarks.part(points[1]).x,
                            facial_lanmarks.part(points[1]).y),
                           (facial_lanmarks.part(points[2]).x,
                            facial_lanmarks.part(points[2]).y),
                           (facial_lanmarks.part(points[3]).x,
                            facial_lanmarks.part(points[3]).y),
                           (facial_lanmarks.part(points[4]).x,
                            facial_lanmarks.part(points[4]).y),
                           (facial_lanmarks.part(points[5]).x,
                            facial_lanmarks.part(points[5]).y)],
                          np.int32)

    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 1)

    cv2.fillPoly(mask, [eye_region], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[0, :]) + 7
    max_y = np.max(eye_region[0, :]) + 10

    gray_eye = left_eye[min_y:max_y, min_x:max_x]

    _, threshold_eye = cv2.threshold(
        gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
    threshold_eye = cv2.resize(threshold_eye, None, fx=7, fy=7)

    t_height, t_width = threshold_eye.shape

    left_side_threshold = threshold_eye[0:t_height, 0: int(t_width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:t_height, int(
        t_width/2): t_width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    gaze_ratio = left_side_white/right_side_white

    return gaze_ratio


while True:

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:

        landmarks = predictor(gray, face)

        le_hor_length, le_ver_length = getBlinkingRatio(
            landmarks, [36, 39, 37, 38, 41, 40])

        re_hor_length, re_ver_length = getBlinkingRatio(
            landmarks, [42, 45, 43, 44, 47, 46])

        ratio = ((le_hor_length + re_hor_length) /
                 (le_ver_length + re_ver_length))

        if ratio > 5.6:
            cv2.putText(frame, "Blinking", (50, 150), font, 3, (255, 0, 0))

        left_eye_gaze_ratio = getGazeRatio(
            landmarks, [36, 37, 38, 39, 40, 41], gray)

        right_eye_gaze_ratio = getGazeRatio(
            landmarks, [42, 43, 44, 45, 46, 47], gray)

        gaze_ratio = (left_eye_gaze_ratio+right_eye_gaze_ratio)/2

        if gaze_ratio >= 1:
            cv2.putText(frame, "Left",
                        (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "right",
                        (50, 100), font, 2, (0, 0, 255), 3)

        # cv2.putText(frame, str(left_eye_gaze_ratio),
            # (50, 100), font, 2, (0, 0, 255), 3)

        # cv2.imshow("Eye", threshold_eye)
        # cv2.imshow("Left Eye", left_side_threshold)
        # cv2.imshow("Right Eye", right_side_threshold)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
