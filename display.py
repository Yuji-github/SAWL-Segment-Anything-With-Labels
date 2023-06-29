"""Display images"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any
import os


def output_images(image: list, selected_masks: List[dict[str, Any]], save: bool = False, file_name: str = None):
    """Display images
    :param image:
        imported_images -> List[np.array]
    :param selected_masks:
    :return:
    """
    if len(selected_masks) == 0:
        return

    def _mapping_masks(selected_masks: List[dict[str, Any]]):
        """Mapping colors with masks for each object
        :param selected_masks:
        :return:
        """

        ax = plt.gca()
        ax.set_autoscale_on(False)

        # create ones array [1, 1, 1, 1] with image height and width
        img = np.ones((selected_masks[0]["segmentation"].shape[0], selected_masks[0]["segmentation"].shape[1], 4))
        img[:, :, 3] = 0  # change the last pos to 0 [1, 1, 1, 0]
        for mask in selected_masks:
            m = mask["segmentation"]  # get mapping masks
            color_mask = np.concatenate([np.random.random(3), [0.35]])  # [rand, rand, rand, 0.35]
            img[m] = color_mask  # boolean array index to give the color for each object (True areas only)
            if "id" in mask:
                plt.text(mask["point_coords"][0][0], mask["point_coords"][0][1], mask["id"], fontsize=46)
        ax.imshow(img)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    _mapping_masks(selected_masks)
    plt.axis("off")

    if save:
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        plt.savefig("outputs/masked_" + file_name)
    else:
        plt.show()
