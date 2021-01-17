# coding=utf-8
import serial
import time

cmd_onekey_fly = 'AAAF50020003AE'  # 一键起飞/降落


def com():
    """搜索串口"""
    name_list = []
    for i in range(30):
        name = 'COM' + str(i)
        try:
            serial.Serial(name)
            name_list.append(name)
        except serial.serialutil.SerialException:
            pass
    print(f"发现{len(name_list)}个串口：", name_list)
    if len(name_list) == 1:
        return name_list[0]
    else:
        if len(name_list) == 0:
            print("请检查遥控器连接")
        else:
            print("请手动选择串口")
        raise SystemExit  # 抛出SystemExit异常，程序退出


# 起降测试
ser = serial.Serial(com(), 500000, timeout=0.5)  # 若报错，请将com()改成所需端口
u_byte = bytes.fromhex(cmd_onekey_fly)  # 字符串形式转为十六进制字节形式
ser.write(u_byte)  # 发送到串口（遥控器）实现一键起飞
time.sleep(3)
ser.write(u_byte)  # 发送到串口 实现一键降落
ser.close()  # 关闭资源
