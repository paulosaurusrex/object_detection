import xml.etree.ElementTree as ET
import os
import torch


def parse_annotation(annotation_path: str,
                     image_dir: str,
                     image_size: tuple[int, int]) -> tuple[list[str], torch.Tensor, list[str]]:
    """
    Traverse the CVAT 1.1 xml tree, get the annotations, and resize them to the scaled image size.

    :param annotation_path: the path of the annotation file.
    :param image_dir: a directory where images in the annotation file are located.
    :param image_size: the size of the images.
    :return: image paths, bounding boxes of relevant objects and their labels.
    """
    image_height, image_width = image_size

    with open(annotation_path, "r") as f:
        tree = ET.parse(f)

    root = tree.getroot()

    image_paths = []
    bounding_boxes = []
    object_labels = []

    for xml_tag in root.findall('image'):
        image_path = os.path.join(image_dir, xml_tag.get('name'))
        image_paths.append(image_path)

        original_image_width = int(xml_tag.get('width'))
        original_image_height = int(xml_tag.get('height'))

        bounding_boxes_in_image = []
        object_labels_in_image = []
        for box in xml_tag.findall('box'):
            x_min = float(box.get('xtl'))  # top left
            y_min = float(box.get('ytl'))
            x_max = float(box.get('xbr'))  # bottom right
            y_max = float(box.get('ybr'))

            # Rescale bounding boxes
            x_min *= image_width / original_image_width
            x_max *= image_width / original_image_width
            y_min *= image_height / original_image_height
            y_max *= image_height / original_image_height
            bounding_box = [x_min, y_min, x_max, y_max]
            bounding_boxes_in_image.append(bounding_box)

            class_label = box.get('label')
            object_labels_in_image.append(class_label)

        bounding_boxes.append(torch.Tensor(bounding_boxes_in_image))
        object_labels.append(object_labels_in_image)

    return image_paths, bounding_boxes, object_labels
