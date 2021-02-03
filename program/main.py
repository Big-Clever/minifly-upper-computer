import cv2
import imutils
import wifi_init
import paddlehub as hub
import os
import time
from multiprocessing import Process, Queue
import uav
import keyboard
import dlib
import numpy as np


def capture(cap_q):
    """图像捕获进程"""
    WIFI = wifi_init.wifi()  # 实例化wifi类
    wifi_init.camera_connect(WIFI)  # 连接飞行器摄像头
    while True:
        wifi_init.check_connection(WIFI)  # 检查wifi连接状态
        start_time = time.time()
        cap = cv2.VideoCapture('http://192.168.1.1:80/snapshot.cgi?user=admin&pwd=')
        if cap.isOpened():
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=640)  # 缩小图片尺寸
            # try:
            #     cap_q.get_nowait()
            # except Exception:pass

            cap_q.put(frame)
            print(f"图像捕获进程接收图片用时{time.time() - start_time}s")


def object_detector(q):
    """物体检测进程"""
    # 加载模型，可选模型 ssd_mobilenet_v1_pascal 和 yolov3_mobilenet_v1_coco2017
    object_detector = hub.Module(name="yolov3_mobilenet_v1_coco2017")
    while True:
        start_time = time.time()  # 记录开始时间
        frame = q.get()
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
        # frame = imutils.resize(frame, width=1280)  # 放大图片尺寸
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break


def uav_control(q):
    """无人机控制进程"""
    minifly = uav.Uav()
    minifly.init_ser()
    print("串口已连接")
    keyboard.hook(minifly.pressed_keys)
    minifly.control_start(q)


def face_detector(frame_queue, face_queue, flag_queue):
    """人脸检测进程"""
    detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
    while True:
        frame = frame_queue.get()
        dets = detector(frame, 1)  # 使用detector进行人脸检测 dets为返回的结果
        face_queue.put(dets)
        flag_queue.put(1)


def show_img(frame_queue, face_queue):
    time_record = time.time()
    while 1:
        frame = frame_queue.get()
        dets = face_queue.get()
        """人脸识别数据处理"""
        print("Number of faces detected: {}".format(len(dets)))  # 打印识别到的人脸个数
        for index, face in enumerate(dets):
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                         face.bottom()))
            # 在图片中标注人脸，并显示
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        fps = 1 / (time.time() - time_record)
        time_record = time.time()
        cv2.putText(frame, f"FPS:{int(fps)} ", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (50, 170, 50), 2)
        # frame = imutils.resize(frame, width=1536)  # 放大图片尺寸
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break


def cap_cache(cap_queue, flag_queue, frame_queue, progress_count):  # 捕获队列，标志队列，图像队列，进程数量
    cache = None
    flag = progress_count
    while 1:
        try:
            cache = cap_queue.get_nowait()
        except Exception:
            pass
        try:
            flag_queue.get_nowait()
            flag += 1
        except Exception:
            pass
        if type(cache) is np.ndarray and flag == progress_count:
            for _ in range(progress_count + 1):
                frame_queue.put(cache)
            cache = None
            flag = 0


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置环境变量
    """父进程创建Queue，并传给各个子进程"""
    cap_queue = Queue(True)  # 捕获图像队列
    ctrl_msg_queue = Queue(True)  # 控制信号队列
    frame_queue = Queue(True)  # 图像队列
    face_queue = Queue(True)  # 人脸识别结果队列
    flag_queue = Queue(True)  # 标志队列
    """进程创建"""
    # Uav_control = Process(target=uav_control, args=(ctrl_msg_queue,))  # 无人机控制进程
    Capture = Process(target=capture, args=(cap_queue,))  # 捕获图像进程
    # Obj_detector = Process(target=object_detector, args=(cap_queue,))  # 物体检测进程
    Face_detector = Process(target=face_detector, args=(frame_queue, face_queue, flag_queue))  # 人脸检测进程
    Show_img = Process(target=show_img, args=(frame_queue, face_queue))  # 图像显示进程
    Cap_cache = Process(target=cap_cache, args=(cap_queue, flag_queue, frame_queue, 1))  # 图像显示进程
    # 启动子进程
    # Uav_control.start()
    Capture.start()
    # Obj_detector.start()
    Face_detector.start()
    Cap_cache.start()
    Show_img.start()

    Capture.join()
