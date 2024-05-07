import torch


def generate_anchor_points(image_size: tuple[int, int]) -> torch.Tensor:
    """
    Generate anchor points based on image size.

    :param image_size: a tuple containing the image height and width.
    :return: a tuple of tensor containing a collection of anchor points x and y coordinates.
    """
    image_height, image_width = image_size

    anchor_xs = torch.arange(0, image_width) + 0.5
    anchor_ys = torch.arange(0, image_height) + 0.5

    return anchor_xs, anchor_ys
