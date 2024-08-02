import cv2

capture = cv2.VideoCapture(0)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    print(frame.shape)
    cv2.imshow("Frame", frame)

capture.release()
cv2.destroyAllWindows()