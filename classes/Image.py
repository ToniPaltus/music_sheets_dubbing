import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from classes.Staff import Staff

class Image:
    """
    Class to work with images.

    :param file_path:
    :type file_path: str
    """

    def __init__(self, file_path: str):
        """
        Constructor.

        :param file_path:
        :type file_path: str
        """

        self.__file_path = file_path
        self.__image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.__height = self.image.shape[0]
        self.__width = self.image.shape[1]
        self.__represents = dict()

        self.__staffs = list()

    def __copy__(self):
        """
        Method to copy object.

        :return:
        """
        return Image(self.file_path)

    @property
    def file_path(self) -> str:
        """
        Getter of file_path.

        :return: file_path
        :rtype: str
        """

        return self.__file_path

    @property
    def image(self) -> np.ndarray:
        """
        Getter of image.

        :return: image
        :rtype: np.ndarray
        """

        return self.__image

    @image.setter
    def image(self, image: np.ndarray) -> None:
        """
        Setter of image.

        :param image: new image
        :type image: np.ndarray
        :type: None
        """

        if image.shape[0] != self.__height or image.shape[1] != self.__width:
            print("Wrong image size")
            return
        self.__image = image

    @property
    def height(self) -> int:
        """
        Getter of height.

        :return: height
        :rtype: int
        """

        return self.__height

    @height.setter
    def height(self, height: int) -> None:
        """
        Setter of height.

        :param height:
        :type height: int
        :rtype: None
        """

        if height <= 0 or not isinstance(height, int):
            print("height must be a positive integer")
            return
        self.__height = height

    @property
    def width(self) -> int:
        """
        Getter of width.

        :return: width
        :rtype: int
        """

        return self.__width

    @width.setter
    def width(self, width: int) -> None:
        """
        Setter of width
        :param width:
        :type: int
        :rtype: None
        """

        if width <= 0 or not isinstance(width, int):
            print("width must be a positive integer")
            return
        self.__width = width

    @property
    def represents(self) -> dict:
        """
        Getter of represents.

        :return: represents
        :rtype: dict
        """

        return self.__represents

    @represents.setter
    def represents(self, represent: dict) -> None:
        """
        Setter of represents.

        :param represent: dict{ key: np.ndarray}
        :type represent: dict
        :rtype: None
        """

        if isinstance(represent, dict):
            self.__represents.update(represent)
            return
        print("It's not a dictionary")

    @property
    def staffs(self) -> list:
        return self.__staffs

    @staffs.setter
    def staffs(self, staffs: list) -> None:
        self.__staffs = staffs

    @staticmethod
    def show_image_plt(image: np.ndarray, name: str) -> None:
        """
        Show image with plt.

        :param image: image
        :type image: np.ndarray
        :param name: plt title name
        :type name: str
        :rtype: None
        """

        plt.imshow(image, cmap="gray")
        plt.title(name)
        plt.show()

    @staticmethod
    def show_image_win(image: np.ndarray, name: str) -> None:
        """
        Show image with win.

        :param image: image
        :type image: np.ndarray
        :param name: plt title name
        :type name: str
        :rtype: None
        """

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def get_concatenated_image(cls, images: list) -> np.ndarray:
        """
        Get concatenated image from the list of images (np.ndarray).
        Need to be the same number of cols and the same depth.

        :param images: list of images
        :type images: list
        :return: image.
        :rtype: np.ndarray
        """
        space_width = 10
        combined_image = images.pop(0)

        for image in images:
            bordered_image = cv2.copyMakeBorder(
                image,
                top=0,
                bottom=0,
                left=0,
                right=space_width,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            combined_image = cv2.hconcat([combined_image, bordered_image])
        return combined_image

    def get_inverted_image(self, image: np.ndarray) -> np.ndarray:
        """
        Get inverted image.

        :param image: image to invert.
        :type image: np.ndarray
        :return: inverted image.
        :rtype: np.ndarray
        """

        self.represents.update({"inverted": cv2.bitwise_not(image)})
        return cv2.bitwise_not(image)

    @staticmethod
    def __rle_encode(arr: np.ndarray) -> tuple:
        """
        Encodes a numpy array using RLE.

        :param arr:
        :type arr: np.ndarray
        :return: count - how many, values - pixel value
        :rtype: tuple
        """

        if len(arr) == 0:
            return [], [], []

        x = np.copy(arr)
        first_mismatch = np.array(x[1:] != x[:-1])
        inequality_positions = np.append(np.where(first_mismatch), len(x) - 1)
        count = np.diff(np.append(-1, inequality_positions))
        values = [x[i] for i in np.cumsum(np.append(0, count))[:-1]]

        return count, values

    def save_image(self) -> None:
        """
        Save the image into it's file_path.

        :rtype: None
        """

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        cv2.imwrite(self.file_path, self.image)

    def get_color_image(self) -> np.ndarray:
        """
        Get the color image.

        :return: image
        :rtype: np.ndarray
        """

        self.represents.update(
            {"color": cv2.imread(self.__file_path, cv2.IMREAD_COLOR)}
        )
        return cv2.imread(self.__file_path, cv2.IMREAD_COLOR)

    def rotate_image(self, angle: int) -> None:
        """
        Anti-clockwise rotate.

        :param angle:
        :type angle: int
        :rtype: None
        """

        center = (self.width // 2, self.height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.__image = cv2.warpAffine(
            self.image, rotation_matrix, (self.width, self.height)
        )

    def resize_image(self, size: tuple) -> None:
        """
        Resize the image with interpolation.

        :param size: (height, width)
        :type size: tuple
        :rtype: None
        """

        self.height = size[0]
        self.width = size[1]
        self.__image = cv2.resize(
            self.image, (self.width, self.height), interpolation=cv2.INTER_CUBIC
        )

    def remove_noise(self) -> None:
        """
        Remove noise.

        :rtype: None
        """

        self.__image = cv2.fastNlMeansDenoising(self.image, None, 10, 7, 21)  # ???

    def get_binarize_image(self, method="otsu") -> np.ndarray:
        """
        Binarize the image 2 methods you choose. Default method is otsu.

        :param method: otsu or adaptive
        :type method: str
        :return: binarize image
        :rtype: np.ndarray
        """

        if method not in ["otsu", "adaptive"]:
            print("Wrong binarize method")
            return self.image

        if method == "otsu":
            retval, binarized_image = cv2.threshold(
                self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self.represents.update({"binarized": binarized_image})
            return binarized_image
        else:
            binarized_image = cv2.adaptiveThreshold(
                self.image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                41,
                10,
            )
            self.represents.update({"binarized": binarized_image})
            return binarized_image

    def rle_axis_encode(self, image: np.ndarray, axis=0) -> tuple:
        """
        Image must be grayscale only. RLE encodes by axis. Default rows.

        :param image: image to encode
        :type image: np.ndarray
        :param axis: axis to encode. 0 - rows, 1 - cols.
        :type axis: int
        :return: count - how many, values - pixel value
        :rtype: tuple
        """

        counts, values = [], []

        if axis == 1:
            for i in range(image.shape[1]):
                col_counts, col_values = self.__rle_encode(image[:, i])
                counts.append(col_counts)
                values.append(col_values)
        else:
            for i in range(image.shape[0]):
                row_counts, row_values = self.__rle_encode(image[i])
                counts.append(row_counts)
                values.append(row_values)

        return counts, values

    def get_rle_sum_data(self, image: np.ndarray) -> np.ndarray:
        """
        Returns sum of rle encoded data.

        :param image: image to get sum_data
        :type image: np.ndarray
        :return: array with sum of black pixels in each row(col)
        :rtype: np.ndarray
        """

        def sum_odd(lst: list) -> int:
            """
            Count sum of elements located in odd positions.

            :param lst: row or col.
            :type lst: list
            :return: sum
            :rtype: int
            """

            return sum(lst[1::2])

        counts, values = self.rle_axis_encode(image)
        sum_data = [sum_odd(lst) for lst in counts]
        sum_data = np.array(sum_data)

        return sum_data

    def get_rle_sum_data_image(self, image: np.ndarray) -> np.ndarray:
        """
        Get rle encoded image.

        :param image: Image to get sum_data_image
        :type image: np.ndarray
        :return: image.
        :rtype: np.ndarray
        """

        height = image.shape[0]
        width = image.shape[1]

        res_image = 255 * np.ones((height, width), dtype=np.uint8)
        sum_data = self.get_rle_sum_data(image)

        for i in range(height):
            res_image[i][: sum_data[i]] = 0

        self.represents.update({"rle_sum_data": res_image})
        return res_image

    def get_Canny_image(self, image: np.ndarray) -> np.ndarray:
        """
        Get Canny edged images.

        :param image: Image to Canny.
        :type image: np.ndarray
        :return: Canny edged image
        :rtype: np.ndarray
        """
        threshold1 = 40
        threshold2 = 100

        self.represents.update({"canny": cv2.Canny(image, threshold1, threshold2)})
        return cv2.Canny(image, threshold1, threshold2)

    def get_horizontal_lines_image(self, image: np.ndarray) -> np.ndarray:
        """
        Get horizontal lines.

        :param image: Image to get horizontal lines
        :type image: np.ndarray
        :return: horizontal lines image
        :rtype: np.ndarray
        """

        minLineLength = 300

        lines = cv2.HoughLinesP(
            image=self.get_Canny_image(image),
            rho=1,
            theta=np.pi / 180,
            threshold=200,
            lines=np.array([]),
            minLineLength=minLineLength,
            maxLineGap=80,
        )

        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 10 or abs(angle - 180) < 10:
                horizontal_lines.append(line)

        res_image = 255 * np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(res_image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

        self.represents.update({"horizontal_lines": res_image})
        return res_image

    def split_into_staffs(self, model):
        result = model(self.image)

        boxes_coords = []
        boxes = result[0].boxes.cpu().numpy()
        for box in boxes:
            xywh = box.xywh

            x = xywh[0, 0]
            y = xywh[0, 1]
            w = xywh[0, 2]
            h = xywh[0, 3]

            boxes_coords.append((x, y, w, h))
        sorted_boxes_coords = sorted(boxes_coords, key=lambda x: x[1])

        for coord in sorted_boxes_coords:
            x, y, w, h = coord

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cropped_staff = Staff(self.image[y1: y2, x1:x2])
            self.__staffs.append(cropped_staff)
