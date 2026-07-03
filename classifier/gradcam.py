"""
gradcam.py
----------
Generate Grad-CAM visualizations for EfficientNet-B0.

Author: NeuroScan Project
"""

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMGenerator:
    """
    Generate Grad-CAM heatmaps for EfficientNet-B0.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained EfficientNet model.
        """

        self.model = model

        # Last convolutional layer of EfficientNet-B0
        self.target_layers = [
            model.features[-1]
        ]

        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

    def generate(
        self,
        input_tensor,
        original_image,
        target_class=None,
        alpha=0.5
    ):
        """
        Generate a Grad-CAM overlay.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape:
                (1, 3, 224, 224)

        original_image : numpy.ndarray
            RGB image with values scaled between 0 and 1.

        target_class : int, optional
            Class index to explain.
            If None, Grad-CAM explains the model's predicted class.

        alpha : float
            Heatmap intensity.
            0.0 = Original image only
            1.0 = Heatmap only

        Returns
        -------
        overlay : numpy.ndarray
            RGB image containing the Grad-CAM overlay.
        """

        # Explain predicted class if none supplied
        targets = None

        if target_class is not None:
            targets = [
                ClassifierOutputTarget(target_class)
            ]

        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        overlay = show_cam_on_image(
            original_image,
            grayscale_cam,
            use_rgb=True,
            image_weight=1 - alpha
        )

        return overlay