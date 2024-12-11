from shurikode.utils import CanvaBit
import numpy as np


class shurikode_encoder:
    def __init__(self, size: int = 1) -> None:
        self.__canvabit = CanvaBit(10, size, 1)
        self.__matrix = np.zeros([10, 10])

    def encode(self, value: int) -> "shurikode_encoder":
        assert value < 256, "The value must be less the n 256."
        bit_array = [0] * 8
        i = -1
        while value:
            bit_array[i] = 1 & value
            value = value >> 1
            i -= 1
        bit_array.reverse()
        bit_array = np.array(bit_array)

        ed_bit = 1 - np.sum(bit_array) % 2

        # Inserting values into matrix
        # Outer cycle
        self.__matrix[0, 1:-1] = bit_array
        self.__matrix[1:-1, -1] = bit_array
        self.__matrix[-1, 1:-1] = bit_array[::-1]
        self.__matrix[1:-1, 0] = bit_array[::-1]

        # Middle cycle
        self.__matrix[1, 2:-1] = 1 - bit_array[:-1]
        self.__matrix[2, 3] = 1 - bit_array[-1]
        self.__matrix[2:-1, -2] = 1 - bit_array[:-1]
        self.__matrix[3, -3] = 1 - bit_array[-1]
        self.__matrix[-2, 1:-2] = 1 - bit_array[0:-1][::-1]
        self.__matrix[-3, -4] = 1 - bit_array[-1]
        self.__matrix[1:-2, 1] = 1 - bit_array[:-1][::-1]
        self.__matrix[-4, 2] = 1 - bit_array[-1]

        # Inner cycle
        self.__matrix[2, 4:8] = bit_array[:-4]
        self.__matrix[3, 4:7] = bit_array[4:-1]
        self.__matrix[4, 4] = bit_array[-1]
        self.__matrix[4:-2, -3] = bit_array[:-4]
        self.__matrix[4:-3, -4] = bit_array[4:-1]
        self.__matrix[4, -5] = bit_array[-1]
        self.__matrix[-3, 2:6] = bit_array[:-4][::-1]
        self.__matrix[-4, 3:6] = bit_array[4:-1][::-1]
        self.__matrix[-5, 5] = bit_array[-1]
        self.__matrix[2:6, 2] = bit_array[:-4][::-1]
        self.__matrix[3:6, 3] = bit_array[4:-1][::-1]
        self.__matrix[5, 4] = bit_array[-1]

        # ED corners
        self.__matrix[0, 0] = ed_bit
        self.__matrix[-1, 0] = ed_bit
        self.__matrix[0, -1] = ed_bit
        self.__matrix[-1, -1] = ed_bit

        return self

    def save(self, path: str):
        self.__canvabit.from_2D_array(self.__matrix).save(path)

    def get_PIL_image(self):
        return self.__canvabit.from_2D_array(self.__matrix).get_PIL_image()
