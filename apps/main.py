import numpy as np


def calc_curvature_coef(x: np.ndarray, y: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Calculate curvature coefficients
    :param x: x coordinates
    :param y: y coordinates
    :param normalize: normalize scale
    :return: curvature coefficients
    """
    x_data = x
    y_data = y
    if normalize:
        x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())
        y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min())

    dx = np.gradient(x_data)
    dy = np.gradient(y_data)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    numerator = dx * ddy - dy * ddx
    denominator = (dx ** 2 + dy ** 2) ** (1.5)

    cc = np.zeros_like(x, dtype=float)
    valid = denominator > 0
    cc[valid] = numerator[valid] / denominator[valid]

    return cc


def closest_color(pixel: np.ndarray | float, color_palette: np.ndarray):  # type: ignore
    # Compute the distance from the pixel to every colour of the palette
    # Hint: colors - pixel gives you all the differences at once
    if pixel.ndim == 0:
        distances = np.abs(pixel - color_palette)
    else:
        distances = np.linalg.norm(pixel - color_palette, axis=1)
    # Return the colour of the palette with the smallest distance
    idx = np.argmin(distances)

    return color_palette[idx]
