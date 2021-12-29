import math
import cv2
import numpy as np


class DrawPose:
    """
    Draw Pose estimation results on canvas.

    Args:
        stickwidth(int): Angle value to draw approximate ellipse curve, default is 4.

    """

    def __init__(self, stickwidth: int = 4):
        self.stickwidth = stickwidth

        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
                        [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                       [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
                       [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                       [255, 0, 170], [255, 0, 85]]

    def __call__(self, canvas: np.ndarray, candidate: np.ndarray, subset: np.ndarray):
        for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                x, y = candidate[index][0:2]
                cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(self.limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), self.stickwidth), \
                                           int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas
