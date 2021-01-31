import cv2
import imutils
import wifi_init
import paddlehub as hub
import os
import time
from multiprocessing import Process, Queue


def capture(q):
    """捕获图像进程"""
    WIFI = wifi_init.wifi()  # 实例化wifi类
    wifi_init.camera_connect(WIFI)  # 连接飞行器摄像头
    while True:
        wifi_init.check_connection(WIFI)  # 检查wifi连接状态
        start_time = time.time()
        cap = cv2.VideoCapture('http://192.168.1.1:80/snapshot.cgi?user=admin&pwd=')
        if cap.isOpened():
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=640)  # 缩小图片尺寸
            if q.empty():
                q.put(frame)
                print(f"进程1接收图片用时{time.time() - start_time}s")


def object_detector(q):
    """物体检测进程"""
    # 加载模型，可选模型 ssd_mobilenet_v1_pascal 和 yolov3_mobilenet_v1_coco2017
    object_detector = hub.Module(name="yolov3_mobilenet_v1_coco2017")
    while True:
        start_time = time.time()  # 记录开始时间
        frame = q.get()
        print(f"进程2等待图片用时{time.time() - start_time}s")
        # if cap.isOpened():
        #     ret, frame = cap.read()
        #     frame = imutils.resize(frame, width=640)  # 缩小图片尺寸
        outputs = object_detector.object_detection(
            images=[frame],
            batch_size=1,
            use_gpu=True,
            score_thresh=0.3,
            visualization=True
        )
        frame = cv2.imread("./detection_result/image_numpy_0.jpg")
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS:{int(fps)} ", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (50, 170, 50), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置环境变量
    cap_queue = Queue()  # 父进程创建Queue，并传给各个子进程
    Capture = Process(target=capture, args=(cap_queue,))
    Obj_detector = Process(target=object_detector, args=(cap_queue,))
    # 启动子进程
    Capture.start()
    Obj_detector.start()
    Capture.join()
    Obj_detector.join()
