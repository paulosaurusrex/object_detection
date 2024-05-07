import torch
import torchvision
import torch.nn as nn


class ResNet50(nn.Module):
    """
    A class that encapsulates the first layers of a pretrained ResNet50 model as backbone for an
    R-CNN models.
    """

    def __init__(self, num_layers: int = 4):
        """
        Creates a ResNet50 backbone for faster R-CNN.

        :param num_layers: the number of initial layers of a ResNet50 to use.
        """
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        layers = list(resnet.children())[:(2 * num_layers)]
        self._model = nn.Sequential(*layers)
        self._unfreeze_parameters()

    def _unfreeze_parameters(self):
        """
        Unfreeze all the parameters of the model, so they can be updated during training.
        """

        for param in self._model.named_parameters():
            param[1].requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward input through the ResNet50 model.

        :param x: input tensor.
        :return: output tensor.
        """
        return self._model(x)
