import torch
from torch.utils.data import Dataset
from object_detection.preprocessing import cvat
import os
from skimage import io
from skimage.transform import resize
from torch.nn.utils.rnn import pad_sequence


class ObjectDetectionDataset(Dataset):
    """
    This class represents a dataset for object detection. I will load images, labels and the
    bounding boxes of relevant objects in the images.
    """

    def __init__(self,
                 annotation_path: str,
                 image_dir: str,
                 image_size: tuple[int, int],
                 class_label_to_index: dict[str, int]):
        """
        Creates a dataset for object detection.

        :param annotation_path: the path of the annotation file.
        :param image_dir: a directory where images in the annotation file are located.
        :param image_size: the size of the images.
        :param class_label_to_index: a dictionary containing a map between class labels and their
            indices.
        """
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.class_label_to_index = class_label_to_index

        self.image_data_, self.bounding_boxes_, self.class_labels_ = self._get_data()

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        :return: the number of images in the dataset.
        """
        return self.image_data_.size(dim=0)

    def __getitem__(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Returns images, bounding boxes of relevant objects and their labels.

        :param indices: indices of the data samples.
        :return:
        """
        return (self.image_data_[indices],
                self.bounding_boxes_[indices],
                self.class_labels_[indices])

    def _get_data(self) -> tuple[torch.Tensor, ...]:
        """
        Loads images, bounding boxes of relevant objects in the images nd their corresponding
            labels.
        :return: a tuple of tensors containing image data, bounding boxes of relevant objects in
            each image and their associated numerical labels.
        """
        image_data = []
        label_indices = []

        image_paths, bounding_boxes, class_labels = cvat.parse_annotation(
            self.annotation_path,
            self.image_dir,
            self.image_size
        )

        for i, image_path in enumerate(image_paths):
            if not image_path or not os.path.exists(image_path):
                continue

            image = io.imread(image_path)
            image = resize(image, self.image_size)

            # Reshape image so channels come first
            image = torch.from_numpy(image).permute(2, 0, 1)

            # Encode labels as integers
            labels_in_image = class_labels[i]
            indices = torch.Tensor([self.class_label_to_index[name] for name in labels_in_image])

            image_data.append(image)
            label_indices.append(indices)

        # Pad bounding boxes and labels such that they are of the same size since each image can
        # have a different number of relevant objects.
        padded_bounding_boxes = pad_sequence(bounding_boxes, batch_first=True, padding_value=-1)
        padded_label_indices = pad_sequence(label_indices, batch_first=True, padding_value=-1)
        image_data = torch.stack(image_data, dim=0)

        return image_data.to(dtype=torch.float32), padded_bounding_boxes, padded_label_indices
