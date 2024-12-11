from PIL import Image, ImageDraw
import numpy as np


class CanvaBit:

    def __init__(self, diagonal: int, size: int = 1, padding: int = 1):
        assert padding == 1 or padding == 0, "The padding must be 0 or 1."
        self.__matrix_diagonal = diagonal
        self.__cell_size = 10 * size
        self.__image_padding = padding * self.__cell_size
        canva_diagonal = diagonal * self.__cell_size + self.__image_padding * 2
        self.__image = Image.new("RGB", (canva_diagonal, canva_diagonal), "white")
        self.__drawable_image = ImageDraw.Draw(self.__image)

    def __draw_bit(self, value: int, x: int, y: int) -> "CanvaBit":
        assert (
            x >= 0
            and x < self.__matrix_diagonal
            and y >= 0
            and y < self.__matrix_diagonal
        ), f"The x and y coordinates must be in the range 0 to {self.__matrix_diagonal} for this canva."
        assert value == 1 or value == 0, "Value must be 0 or 1."
        color = "black" if value == 1 else "white"
        x0 = self.__image_padding + x * self.__cell_size
        y0 = self.__image_padding + y * self.__cell_size
        x1 = x0 + self.__cell_size
        y1 = y0 + self.__cell_size
        self.__drawable_image.rectangle([x0, y0, x1, y1], fill=color)
        return self

    def from_2D_array(self, a: np.ndarray) -> "CanvaBit":
        assert len(a.shape) == 2, "The provided array must be with two dimensions."
        rows, cols = a.shape
        for i in range(rows):
            for j in range(cols):
                self.__draw_bit(a[i][j], j, i)
        return self

    def save(self, path: str):
        self.__image.save(path)

    def get_PIL_image(self):
        return self.__image
