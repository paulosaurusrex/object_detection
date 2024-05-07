from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import ops
import matplotlib.patches as patches


def display_image_batch(image_batch: torch.Tensor | np.ndarray, axs: list[plt.Axes]):
    """
    Displays a collection of images in a list of axes.

    :param image_batch: a batch of images.
    :param axs: axes where to plot the images.
    """
    for i, image in enumerate(image_batch):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        axs[i].imshow(image)


def display_bounding_boxes(
        bounding_boxes: torch.Tensor | np.ndarray,
        ax: plt.Axes,
        class_labels: list[str] = None,
        in_format: str = 'xyxy',
        color: str = 'y',
        line_width: float = 3):
    """
    Displays a collection of bounding boxes in a list of axes.

    :param bounding_boxes: a list of bounding boxes.
    :param ax: axis where to plot the bounding boxes.
    :param class_labels: an optional list of class labels to display inside each bounding box.
    :param in_format: either 'xyxy' or 'xywh'
    :param color: the color of the bounding boxes.
    :param line_width: the width of the lines in the bounding boxes.
    """
    if isinstance(bounding_boxes, np.ndarray):
        bounding_boxes = torch.from_numpy(bounding_boxes)

    if class_labels and len(class_labels) != len(bounding_boxes):
        raise Exception(f'The number of elements in the label batch ({len(class_labels)}) '
                        f'differs from the number of elements in the bounding box batch '
                        f'({len(bounding_boxes)}).')

    # Convert boxes to x, y, width, height format
    bounding_boxes = ops.box_convert(bounding_boxes, in_fmt=in_format, out_fmt='xywh')
    for i, box in enumerate(bounding_boxes):
        x, y, w, h = box.numpy()

        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=line_width,
                                 edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)

        if class_labels:
            if class_labels[i] == 'pad':
                continue
            ax.text(x + 5, y + 20, class_labels[i], bbox=dict(facecolor='yellow', alpha=0.5))


def display_grid(x_points: torch.Tensor | np.ndarray,
                 y_points: torch.Tensor | np.ndarray,
                 ax: plt.Axes,
                 special_point: Optional[tuple[int, int]] = None):
    """
    Displays a grid of points.

    :param x_points: x coordinates of the points.
    :param y_points: y coordinates of the points.
    :param ax: axis where to plot the points.
    :param special_point: an optional special point we want to emphasize on the grid.
    """
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')

    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
