import cv2
import numpy as np
from MiDaS import depth_utils


def nothing(x):
    pass


class FakeDeFocus:
    def __init__(self, image_path):
        depth_utils.load_model()

        self.image = cv2.imread(image_path)
        self.image = cv2.pyrDown(self.image)
        self.depth = depth_utils.predict_depth(self.image)

    @staticmethod
    def normalize01(x, axis=None):
        return (x - np.min(x, axis=axis)) / (np.max(x, axis=axis) - np.min(x, axis=axis))

    def create(self, x):
        trackbar_pos = cv2.getTrackbarPos("r", "Result") - 200
        depth = cv2.cvtColor(self.depth, cv2.COLOR_GRAY2BGR)
        depth_float = depth / 255.
        if trackbar_pos < 0:
            depth_float = 1. - depth_float
            trackbar_pos = 200 + trackbar_pos
        depth_float **= trackbar_pos / 100.
        blur = cv2.GaussianBlur(self.image, (11, 11), cv2.BORDER_DEFAULT)
        return np.uint8(depth_float * self.image + (1. - depth_float) * blur)


if __name__ == '__main__':
    fdf = FakeDeFocus(image_path='./images/girona-4278090_1920.jpg')

    cv2.namedWindow("Result")
    cv2.createTrackbar("r", "Result", 200, 400, nothing)

    cv2.imshow("Original", fdf.image)
    cv2.imshow("Depth", fdf.depth)

    while True:
        result = fdf.create(None)

        cv2.imshow("Result", result)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
