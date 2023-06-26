import numpy as np


class ClusterImages:
    def __init__(self, image: np.array, masks: dict):
        self.image = image
        self.sorted_area_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        self.seg_images: list = []

    def extract_masked_images(self) -> None:
        for mask in self.sorted_area_masks:
            self.seg_images.append(self.image * mask["segmentation"])
