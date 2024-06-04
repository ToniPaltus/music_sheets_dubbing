import numpy as np
import cv2
from matplotlib import pyplot as plt

class Staff:
    def __init__(self, staff: np.ndarray):
        self.__staff = staff

    @property
    def staff(self):
        return self.__staff

    @staff.setter
    def staff(self, staff: np.ndarray):
        self.__staff = staff

    def save_staff_to_dir(self, path: str, name: str) -> None:
        cv2.imwrite(path + name + '.jpg', self.staff)

    def show_win(self, name: str) -> None:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, self.staff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_plt(self, name: str) -> None:

        plt.imshow(self.staff, cmap="gray")
        plt.title(name)
        plt.show()
