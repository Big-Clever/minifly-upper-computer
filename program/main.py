import cv2
import imutils
from wifi_init import camera_connect

# 连接飞行器摄像头
camera_connect()

url = 'http://192.168.1.1:80/snapshot.cgi?user=admin&pwd='

cnt = 0
while True:
    timer = cv2.getTickCount()
    cap = cv2.VideoCapture(url)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if cap.isOpened():
        cnt += 1
        width, height = cap.get(3), cap.get(4)
        # print(cnt, '[', width, height, ']')
        ret, frame = cap.read()
        # frame = imutils.resize(frame, width=640)
        # frame = cv2.flip(frame, -180)
        # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow('frame', frame)
    else:
        print("Error")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()