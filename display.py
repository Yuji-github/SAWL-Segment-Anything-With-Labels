"""Display images"""

import matplotlib.pyplot as plt
import numpy as np

def output_images(image:list, masks:dict):
    """Display images

    :param image:
        imported_images -> List[np.array]
    :param masks:
    :return:
    """
    def _mapping_masks(masks: dict):
        """Mapping colors with masks for each object

        :param masks:
        :return:
        """
        if len(masks) == 0:
            return
        # sorting masks with area sizes (big -> small)
        sorted_area_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

        ax = plt.gca()
        ax.set_autoscale_on(False)

        # create ones array [1, 1, 1, 1] with image height and width
        img = np.ones((sorted_area_masks[0]["segmentation"].shape[0], sorted_area_masks[0]["segmentation"].shape[1], 4))
        img[:, :, 3] = 0  # change the last pos to 0 [1, 1, 1, 0]
        for mask in sorted_area_masks:
            m = mask["segmentation"]  # get mapping masks
            color_mask = np.concatenate([np.random.random(3), [0.35]])  # [rand, rand, rand, 0.35]
            img[m] = color_mask  # boolean array index to give the color for each object (True areas only)
        ax.imshow(img)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    _mapping_masks(masks)
    plt.axis("off")
    plt.show()