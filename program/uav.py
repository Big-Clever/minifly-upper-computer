import serial
import time
import struct
import keyboard


class Uav(object):
    def __init__(self):
        self.com = None
        self.ser = None
        self.key_ctrl_times = 0
        self.auto_ctrl_times = 0
        self.key_data = [0] * 6
        self.auto_data = [0] * 6
        # self.real_data = [0, 0, 0, 50]
        self.SPEED_YAW = 100
        self.SPEED_THRUST = 50
        self.SPEED_PITCH = 10
        self.SPEED_ROLL = 10

    def search_com(self):
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
            self.com = name_list[0]

    def init_ser(self):
        while not self.com:
            self.search_com()
            time.sleep(0.5)
        self.ser = serial.Serial(self.com, 500000, timeout=0.5)

    def send_com(self, s_str):
        h_str = bytes.fromhex(s_str)
        self.ser.write(h_str)

    @staticmethod
    def float_to_hex(data):
        return (struct.pack('<f', data)).hex()

    def cmd_data_send(self, flydata):
        dataHead = "AAAF501D01"  # 1D代表datalen+1, 01代表控制数据
        strTrim = "0000000000000000"
        strMode = "01000000"  # 前两位00手动 01定高
        cmd_text = dataHead + self.float_to_hex(flydata[0]) + self.float_to_hex(flydata[1]) \
                   + self.float_to_hex(flydata[2]) + self.float_to_hex(flydata[3]) + strTrim + strMode
        h_byte = bytes.fromhex(cmd_text)
        checksum = 0
        for a_byte in h_byte:
            checksum += a_byte
        H = hex(checksum % 256)
        if H[-2] == 'x':  # 0xF -> 0x0F
            cmd_text = cmd_text + '0' + H[-1]
        else:
            cmd_text = cmd_text + H[-2] + H[-1]
        self.send_com(cmd_text)

    def take_off(self):  # 起飞/降落
        data = 'AAAF5003000300AF'
        self.send_com(data)

    def flip(self, direction):  # 翻滚 direction:CENTER=0,FORWARD,BACK,LEFT,RIGHT
        if direction == 1:
            data = 'AAAF5003000501B2'
        elif direction == 2:
            data = 'AAAF5003000502B3'
        elif direction == 3:
            data = 'AAAF5003000503B4'
        elif direction == 4:
            data = 'AAAF5003000504B5'
        else:
            return
        self.send_com(data)

    def pressed_keys(self, e):
        if keyboard._pressed_events:  # 判断字典非空
            data = [0] * 6
            if 17 in keyboard._pressed_events:  # w
                data[1] += 1
            if 31 in keyboard._pressed_events:  # s
                data[1] -= 1
            if 30 in keyboard._pressed_events:  # a
                data[0] -= 1
            if 32 in keyboard._pressed_events:  # d
                data[0] += 1
            if 72 in keyboard._pressed_events:  # up
                data[3] += 1
            if 80 in keyboard._pressed_events:  # down
                data[3] -= 1
            if 16 in keyboard._pressed_events:  # q
                data[2] -= 1
            if 18 in keyboard._pressed_events:  # e
                data[2] += 1
            if 75 in keyboard._pressed_events:  # left
                data[2] -= 1
            if 77 in keyboard._pressed_events:  # right
                data[2] += 1
            if 57 in keyboard._pressed_events:  # space
                data[4] = 1
            if 28 in keyboard._pressed_events:  # enter
                data[5] = 1
            self.key_data = data
            if self.key_ctrl_times == 0:
                self.key_ctrl_times = 250
            else:
                self.key_ctrl_times = 20  # 手动控制次数
            self.auto_ctrl_times = 0  # 自动控制次数清零

    @staticmethod
    def limit(value, value_limit):
        if value > value_limit:
            value = value_limit
        elif value < -value_limit:
            value = value_limit
        return value

    def control_start(self, q=None):
        while 1:
            try:
                self.auto_data = q.get_nowait()
                self.auto_ctrl_times = 150
            except Exception:
                self.auto_data = None

            if self.key_ctrl_times > 0:  # 执行按键控制
                if self.key_data[4] == 1:
                    self.take_off()  # 起飞/降落
                    self.key_ctrl_times = 0  # 清除控制次数
                    print("起飞")
                elif self.key_data[5] > 0:
                    self.flip(self.key_data[5])  # 空翻
                    self.key_ctrl_times = 0  # 清除控制次数
                    print("空翻")
                else:
                    data = self.key_data[0:4]
                    data = [data[0] * 10, data[1] * 10, data[2] * 100, data[3] * 50 + 50]
                    self.cmd_data_send(data)  # 发送控制数据
                    self.key_ctrl_times -= 1  # 控制次数-1
            elif self.auto_ctrl_times > 0:  # 执行自动控制
                if self.auto_data[4] == 1:
                    self.take_off()  # 起飞/降落
                    self.auto_ctrl_times = 0  # 清除控制次数
                    print("起飞")
                elif self.auto_data[5] > 0:
                    self.flip(self.auto_data[5])  # 空翻
                    self.auto_ctrl_times = 0  # 清除控制次数
                    print("空翻")
                else:
                    data = self.auto_data[0:4]
                    data[0] = self.limit(data[0], self.SPEED_ROLL)
                    data[1] = self.limit(data[1], self.SPEED_PITCH)
                    data[2] = self.limit(data[2], self.SPEED_YAW)
                    data[3] = self.limit(data[3], self.SPEED_THRUST) + 50
                    self.cmd_data_send(data)  # 发送控制数据
                    self.auto_ctrl_times -= 1  # 控制次数-1
            time.sleep(0.001)


if __name__ == '__main__':
    uav = Uav()
    uav.init_ser()
    print("串口已连接")
    keyboard.hook(uav.pressed_keys)
    uav.control_start()
