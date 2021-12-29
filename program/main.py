import cv2
import imutils
import wifi_init
import paddlehub as hub
import os
import time
from multiprocessing import Process, Queue
import uav
import keyboard
import numpy as np
import pid
import deepsort


def uav_control(q):
    """无人机控制进程"""
    minifly = uav.Uav()
    minifly.init_ser()
    print("串口已连接")
    keyboard.hook(minifly.pressed_keys)
    minifly.control_start(q)


def cap_cache(cap_queue, flag_queue, frame_queue, progress_count):  # 捕获队列，标志队列，图像队列
    """图像分配进程"""
    cache = None
    flag = 1
    start_time = time.perf_counter()
    while 1:
        try:
            cache = cap_queue.get_nowait()
        except Exception:pass
        try:
            flag_queue.get_nowait()
            flag = 1
            start_time = time.perf_counter()
        except Exception:pass
        if flag == 1 and type(cache) is np.ndarray:
            print(f"2图像分配进程等待接收图片时间：{time.perf_counter()-start_time}s")
            for _ in range(progress_count + 1):
                frame_queue.put(cache)
            cache = None
            flag = 0


def capture(cap_q):
    """图像捕获进程"""
    start_time = time.perf_counter()
    WIFI = wifi_init.wifi()  # 实例化wifi类
    wifi_init.camera_connect(WIFI)  # 连接飞行器摄像头
    while time.perf_counter()-start_time < 5:  # 等待其他进程初始化
        time.sleep(0.5)
    while True:
        start_time = time.perf_counter()
        wifi_init.check_connection(WIFI)  # 检查wifi连接状态
        cap = cv2.VideoCapture('http://192.168.1.1:80/snapshot.cgi?user=admin&pwd=')
        if cap.isOpened():
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=640)  # 缩小图片尺寸
            cap_q.put(frame)
            print(f"1捕获进程接收图片时间:{time.perf_counter() - start_time}s")


def object_detector(frame_queue, obj_queue):
    """物体检测进程"""
    # 加载模型，可选模型 ssd_mobilenet_v1_pascal 和 yolov3_mobilenet_v1_coco2017
    object_detector = hub.Module(name="yolov3_mobilenet_v1_coco2017")
    start_time = time.perf_counter()
    while True:
        frame = frame_queue.get()
        start_time = time.perf_counter()
        outputs = object_detector.object_detection(
            images=[frame],
            batch_size=1,
            use_gpu=True,
            score_thresh=0.4,
            visualization=False
        )
        obj_queue.put(outputs)
        print(f"3物体检测完成用时：{time.perf_counter() - start_time}s")
        start_time = time.perf_counter()


def face_detector(frame_queue, face_queue):
    """人脸检测进程"""
    face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
    while True:
        frame = frame_queue.get()
        start_time = time.perf_counter()
        result = face_detector.face_detection(
            images=[frame],
            use_gpu=False,
            visualization=False,
            confs_threshold=0.8)
        face_queue.put(result)
        print(f"4人脸检测完成用时：{time.perf_counter() - start_time}s")


def human_track(frame_queue, target_queue):
    """目标跟踪进程"""
    human_track = deepsort.DeepSort(use_gpu=True, threshold=0.2)
    while 1:
        frame = frame_queue.get()
        start_time = time.perf_counter()
        outputs = human_track.update(frame)
        target_queue.put(outputs)
        print(f"5目标跟踪完成用时：{time.perf_counter() - start_time}s")


def pose_estimation(frame_queue, pose_queue):
    """姿态检测进程"""
    # 可选模型 openpose_body_estimation 和 openpose_hands_estimation
    pose_estimation = hub.Module(name="openpose_body_estimation")
    print(f"6姿态检测准备就绪")
    while True:
        frame = frame_queue.get()
        print(f"6姿态检测已接受图像")
        start_time = time.perf_counter()
        outputs = pose_estimation.predict(
            img=frame,
            # scale=[0.5, 1.0, 1.5, 2.0],  # 识别手部关键点时,使用图片的不同尺度
            visualization=False)
        pose_queue.put(outputs)
        print(f"6姿态检测完成用时：{time.perf_counter() - start_time}s")


def depth_estimation(frame_queue, depth_queue):
    """深度估计进程"""
    model = hub.Module(name='MiDaS_Small', use_gpu=True)
    while True:
        frame = frame_queue.get()
        start_time = time.perf_counter()
        outputs = model.depth_estimation(
            images=[frame],
            paths=None,
            batch_size=1)
        depth = outputs[0]
        # depth_min = depth.min()
        # depth_max = depth.max()
        # max_val = 2 ** 8 - 1
        # if depth_max - depth_min > np.finfo("float").eps:
        #     out = max_val * (depth - depth_min) / (depth_max - depth_min)
        # else:
        #     out = np.zeros(depth.shape, dtype=depth.type)
        # depth_queue.put(out.astype("uint8"))
        depth_queue.put(depth)
        print(f"7深度估计完成用时：{time.perf_counter() - start_time}s")


def show_img(frame_queue, flag_queue, ctrl_msg_queue, obj_queue, face_queue, target_queue, pose_queue, depth_queue):
    """图像显示进程"""
    Roll_pid = pid.PID(0.017, 0.002, 0.01, 0.1)  # 0.017, 0.002, 0.02, 0.1
    Pitch_pid = pid.PID(4, 0.2, 4, 0.3)  # 4, 0.2, 4, 0.3
    Height_pid = pid.PID(0.2, 0, 0.0005, 0)  # 0.2, 0, 0.0005, 0
    Yaw_pid = pid.PID(0.3, 0, 0.02, 0)  # 0.3, 0, 0.02, 0
    face_size = 250  # 若设为0，则为第一帧图像所测距离
    pitch_err = 0
    target = -1  # 目标序号
    start_time = time_record = time.perf_counter()
    while 1:
        frame = frame_queue.get()
        print(f"8图像显示进程等待{time.perf_counter()-start_time}s")
        start_time = time.perf_counter()
        """深度估计数据处理"""
        # depth_ret = depth_queue.get()
        # red_depth = np.where(depth_ret>700,200,0).astype("uint8")
        # frame[:, :, 2] = cv2.add(frame[:, :, 2], red_depth)
        """物体检测数据处理"""
        # obj_res = obj_queue.get()
        # for data in obj_res[0]["data"]:
        #     left = int(data["left"])
        #     top = int(data["top"])
        #     right = int(data["right"])
        #     bottom = int(data["bottom"])
        #     cv2.rectangle(frame, (left, top), (right, bottom), (255, 100, 100), 2)
        #     cv2.putText(frame, f"{data['label']}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (43, 43, 43), 1)
        """人脸识别数据处理"""
        # face_ret = face_queue.get()
        # for face in face_ret[0]["data"]:
        #     left = int(face["left"])
        #     top = int(face["top"])
        #     right = int(face["right"])
        #     bottom = int(face["bottom"])
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)   # 在图片中标注人脸，并显示
        """姿态检测数据处理"""
        pose_ret = pose_queue.get()
        frame = pose_ret["data"]
        """目标跟踪数据处理"""
        outputs = target_queue.get()
        if outputs is not None:
            for index, output in enumerate(outputs):
                cv2.rectangle(frame, (output[0], output[1]), (output[2], output[3]), (0, 0, 255), 2)
                cv2.putText(frame, str(output[-1]), (output[0], output[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                if output[-1]:
                    target = index
        """识别结果接收完成，标志位置1"""
        # flag_queue.put(1)
        """自动飞行控制"""
        if target >= 0:
            left = outputs[target][0]
            top = outputs[target][1]
            right = outputs[target][2]
            bottom = outputs[target][3]
            target = -1
            "水平方向"
            center_x = frame.shape[1]/2
            x_err = (right + left)/2 - center_x
            roll_ctrl = Roll_pid.update(x_err)
            "垂直方向"
            center_z = frame.shape[0] / 2
            z_err = (bottom + top)/2 - center_z
            height_ctrl = Height_pid.update(-z_err)
            "画中心点"
            cv2.circle(frame, (int(center_x), int(center_z)), 1, (0, 0, 255), 0 )
            "前后方向"
            if face_size == 0:
                face_size = right - left
                print("face_size=",face_size)
                pitch_err = 0
            else:
                pitch_err = (face_size/(right - left) - 1) * 0.3 + pitch_err * 0.7  # 滑动滤波
                print("face_size=", right - left)
            pitch_ctrl = Pitch_pid.update(pitch_err) + 0.25
            "旋转角度"
            yaw_ctrl = Yaw_pid.update(x_err)

            print("=" * 80)
            print([roll_ctrl, pitch_ctrl, yaw_ctrl, height_ctrl])
            print("=" * 80)
            ctrl_msg_queue.put([roll_ctrl, pitch_ctrl, 0, height_ctrl, 0, 0])  # 水平跟随
            # ctrl_msg_queue.put([0, pitch_ctrl, yaw_ctrl, height_ctrl, 0, 0])  # 旋转跟随
        """图像显示"""
        fps = 1 / (time.perf_counter() - time_record)
        time_record = time.perf_counter()
        cv2.putText(frame, f"FPS:{int(fps)} ", (550, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # frame = imutils.resize(frame, width=1536)  # 放大图片尺寸
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break
        print(f"8图像显示用时：{time.perf_counter() - start_time}s")
        start_time = time.perf_counter()
        flag_queue.put(1)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置环境变量
    img_process_num = 2  # 图像处理进程数量，包括：物体检测、人脸检测、人体跟踪、姿态识别、深度估计
    """父进程创建Queue，并传给各个子进程"""
    cap_queue = Queue(True)  # 捕获图像队列
    ctrl_msg_queue = Queue(True)  # 控制信号队列
    frame_queue = Queue(True)  # 图像队列
    # show_queue = Queue(True)  # 显示图像队列
    obj_queue = Queue(True)  # 物体检测结果队列
    face_queue = Queue(True)  # 人脸识别结果队列
    target_queue = Queue(True)  # 目标跟踪结果队列
    pose_queue = Queue(True)  # 姿态检测结果队列
    depth_queue = Queue(True)  # 深度估计结果队列
    flag_queue = Queue(True)  # 标志队列
    """进程创建"""
    Uav_control = Process(target=uav_control, args=(ctrl_msg_queue,))  # 无人机控制进程
    Capture = Process(target=capture, args=(cap_queue,))  # 捕获图像进程
    Cap_cache = Process(target=cap_cache, args=(cap_queue, flag_queue, frame_queue, img_process_num))  # 图像分配进程
    Obj_detector = Process(target=object_detector, args=(frame_queue, obj_queue))  # 物体检测进程
    Face_detector = Process(target=face_detector, args=(frame_queue, face_queue))  # 人脸检测进程
    Human_track = Process(target=human_track, args=(frame_queue, target_queue))  # 人类跟踪进程
    Pose_estimation = Process(target=pose_estimation, args=(frame_queue, pose_queue))  # 姿态检测进程
    Depth_estimation = Process(target=depth_estimation, args=(frame_queue, depth_queue))  # 深度估计进程
    Show_img = Process(target=show_img, args=(frame_queue, flag_queue, ctrl_msg_queue, obj_queue, face_queue, target_queue, pose_queue, depth_queue))  # 图像显示进程
    """启动子进程"""
    Uav_control.start()  # 无人机控制进程
    Capture.start()  # 捕获图像进程
    Show_img.start()  # 图像分配进程
    # Obj_detector.start()  # 物体检测进程
    # Face_detector.start()  # 人脸检测进程
    Human_track.start()  # 人类跟踪进程
    Pose_estimation.start()  # 姿态检测进程
    # Depth_estimation.start()  # 深度估计进程
    Cap_cache.start()  # 图像显示进程


    Capture.join()
