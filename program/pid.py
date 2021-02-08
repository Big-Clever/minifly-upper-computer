import time


class PID:
    """PID控制器"""

    def __init__(self, p, i, d, windup):

        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.windup_guard = windup
        self.ITerm = 0

        self.last_time = 0
        self.last_error = 0

    def update(self, error):
        """计算给定参考反馈的PID值
           公式:u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        current_time = time.time()
        delta_time = current_time - self.last_time
        delta_error = error - self.last_error

        self.last_time = current_time
        self.last_error = error

        PTerm = self.Kp * error

        if delta_time > 0.5:  # 若与上次控制时间相差0.5秒以上，则只返回比例项
            return PTerm

        self.ITerm += self.Ki * error * delta_time

        if self.ITerm < -self.windup_guard:
            self.ITerm = -self.windup_guard
        elif self.ITerm > self.windup_guard:
            self.ITerm = self.windup_guard

        if delta_time > 0:
            DTerm = delta_error / delta_time
        else:
            DTerm = 0

        output = PTerm + self.ITerm + (self.Kd * DTerm)
        print(error, PTerm, self.Ki * self.ITerm, self.Kd * DTerm, output)
        return output

    def set_param(self, p, i, d, windup):
        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.windup_guard = windup
