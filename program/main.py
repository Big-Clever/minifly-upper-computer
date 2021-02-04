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


def uav_control(q):
    """无人机控制进程"""
    minifly = uav.Uav()
    minifly.init_ser()
    print("串口已连接")
    keyboard.hook(minifly.pressed_keys)
    minifly.control_start(q)


def capture(cap_q):
    """图像捕获进程"""
    WIFI = wifi_init.wifi()  # 实例化wifi类
    wifi_init.camera_connect(WIFI)  # 连接飞行器摄像头
    while True:
        wifi_init.check_connection(WIFI)  # 检查wifi连接状态
        cap = cv2.VideoCapture('http://192.168.1.1:80/snapshot.cgi?user=admin&pwd=')
        if cap.isOpened():
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=640)  # 缩小图片尺寸
            cap_q.put(frame)
            print(f"1捕获进程接收图片时间{time.time()}")


def object_detector(frame_queue, obj_queue, flag_queue):
    """物体检测进程"""
    # 加载模型，可选模型 ssd_mobilenet_v1_pascal 和 yolov3_mobilenet_v1_coco2017
    object_detector = hub.Module(name="yolov3_mobilenet_v1_coco2017")
    flag_queue.put(1)
    while True:
        frame = frame_queue.get()
        print(f"3物体检测进程接收时间：{time.time()}")
        outputs = object_detector.object_detection(
            images=[frame],
            batch_size=1,
            use_gpu=True,
            score_thresh=0.3,
            visualization=False
        )
        obj_queue.put(outputs)
        flag_queue.put(1)


def face_detector(frame_queue, face_queue, flag_queue):
    """人脸检测进程"""
    detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
    flag_queue.put(1)
    while True:
        frame = frame_queue.get()
        print(f"4人脸检测进程接收时间：{time.time()}")
        dets = detector(frame, 1)  # 使用detector进行人脸检测 dets为返回的结果
        face_queue.put(dets)
        flag_queue.put(1)


def show_img(frame_queue, obj_queue, face_queue):
    """图像显示进程"""
    time_record = time.time()
    while 1:
        frame = frame_queue.get()
        print(f"8图像显示进程接收{time.time()}")
        """物体检测数据处理"""
        obj_res = obj_queue.get()
        print(obj_res)
        for data in obj_res[0]["data"]:
            left = int(data["left"])
            top = int(data["top"])
            right = int(data["right"])
            bottom = int(data["bottom"])
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 100, 100), 2)
            cv2.putText(frame, f"{data['label']}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (43, 43, 43), 1)
        """人脸识别数据处理"""
        dets = face_queue.get()
        print("Number of faces detected: {}".format(len(dets)))  # 打印识别到的人脸个数
        for index, face in enumerate(dets):
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                         face.bottom()))
            # 在图片中标注人脸，并显示
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        """图像显示"""
        fps = 1 / (time.time() - time_record)
        time_record = time.time()
        cv2.putText(frame, f"FPS:{int(fps)} ", (550, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)
        # frame = imutils.resize(frame, width=1536)  # 放大图片尺寸
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break


def cap_cache(cap_queue, flag_queue, frame_queue, progress_count):  # 捕获队列，标志队列，图像队列，进程数量
    """图像分配进程"""
    cache = None
    flag = 0
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
            print(f"2图像分配进程发送图片时间：{time.time()}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置环境变量
    img_process_num = 2  # 图像处理进程数量，包括：物体检测、人脸检测、人体跟踪、姿态识别、深度估计
    """父进程创建Queue，并传给各个子进程"""
    cap_queue = Queue(True)  # 捕获图像队列
    ctrl_msg_queue = Queue(True)  # 控制信号队列
    frame_queue = Queue(True)  # 图像队列
    obj_queue = Queue(True)  # 物体检测结果队列
    face_queue = Queue(True)  # 人脸识别结果队列
    flag_queue = Queue(True)  # 标志队列
    """进程创建"""
    Uav_control = Process(target=uav_control, args=(ctrl_msg_queue,))  # 无人机控制进程
    Capture = Process(target=capture, args=(cap_queue,))  # 捕获图像进程
    Obj_detector = Process(target=object_detector, args=(frame_queue, obj_queue, flag_queue))  # 物体检测进程
    Face_detector = Process(target=face_detector, args=(frame_queue, face_queue, flag_queue))  # 人脸检测进程
    Show_img = Process(target=show_img, args=(frame_queue, obj_queue, face_queue))  # 图像显示进程
    Cap_cache = Process(target=cap_cache, args=(cap_queue, flag_queue, frame_queue, img_process_num))  # 图像分配进程
    # 启动子进程
    Uav_control.start()  # 无人机控制进程
    Capture.start()  # 捕获图像进程
    Show_img.start()  # 图像分配进程
    Obj_detector.start()  # 物体检测进程
    Face_detector.start()  # 人脸检测进程
    Cap_cache.start()  # 图像显示进程


    Capture.join()
