import cv2
import imutils
import wifi_init

# 连接飞行器摄像头
WIFI = wifi_init.wifi()  # 实例化wifi类
wifi_init.camera_connect(WIFI)

cap = cv2.VideoCapture('http://192.168.1.1:80/snapshot.cgi?user=admin&pwd=')

cnt = 0
while True:
    wifi_init.check_connection(WIFI)  # 检查wifi连接状态
    timer = cv2.getTickCount()
    cap.open('http://192.168.1.1:80/snapshot.cgi?resolution=11&user=admin&pwd=')
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if cap.isOpened():
        cnt += 1
        width, height = cap.get(3), cap.get(4)
        print(cnt, '[', width, height, ']')
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow('frame', frame)
    else:
        print("Error")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
