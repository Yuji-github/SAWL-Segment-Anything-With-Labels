from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import argparse
import os


def parse_args() -> argparse:
    parser = argparse.ArgumentParser()

    # import images and target segmentation names
    parser.add_argument(
        "--image_folder_path", "-if", type=str, default="images", help="folder path for images", required=True
    )
    parser.add_argument("--target_list", "-tl", nargs="+", help="target list for segmentation", required=True)

    # setting SAM
    parser.add_argument("--model_type", "-mt", type=str, default="vit_h", help="model typs for SAM")
    parser.add_argument(
        "--checkpoint",
        "-cp",
        type=str,
        default="model_checkpoint/sam_vit_h_4b8939.pth",
        help="checkpoint: more info models_checkpoint/download_link_for_chec_kpoint.txt",
    )

    # select generate masks or predict with prompts
    parser.add_argument(
        "--generate_mask",
        "-gm",
        type=eval,
        choices=[True, False],
        default=True,
        help="generating segmentation masks with SAM",
        required=True,
    )

    return parser.parse_args()


def _import_images(images: list) -> List[cv2]:
    """Importing images to cv2, convert the images to RGB
    SAM requires HWC uint8 format

    :param images:
    :return List[cv2]:
        RGB images in the list
    """
    if not images:
        return []

    imported_images = []
    for image in images:
        imported_images.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    return imported_images


def _set_up_SAM_mask_generator(sam: sam_model_registry = None, **kwargs) -> SamAutomaticMaskGenerator:
    """Setting SAM mask generator
    Assume users do not give input prompts such as point is in (X,Y) in pixels

    :param sam:
        SAM model
    :param kwargs:
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode
    :return SamAutomaticMaskGenerator:
    """
    return SamAutomaticMaskGenerator(model=sam, **kwargs)


def _set_up_SAM_predict_with_prompt(sam: sam_model_registry = None) -> SamPredictor:
    """Setting SAM predictor
    User must know prompts of each image

    prompts: *NOT a word such as "Cat"
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    box: Optional[np.ndarray] = None,
    mask_input: Optional[np.ndarray] = None,
    multimask_output: bool = True,
    return_logits: bool = False

    :param sam:
        SAM model
    :return SamPredictor:
    """
    return SamPredictor(sam)


def _show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


if __name__ == "__main__":
    args = parse_args()

    images = dir_list = os.listdir(args.image_folder_path)
    imported_images = _import_images(images)

    # SAM registry
    # model_type: recommend vit_h (2.4 GB) to get more segmentation masks
    # checkpoint: this model must match with the model_types
    # if GPU available, put SAM on GPU otherwise CPU (CPU takes time)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")

    if args.generate_mask:  # generate masks (default)
        # todo: get kwags from args
        sam = _set_up_SAM_mask_generator(sam=sam)
    else:  # predict masks
        sam = _set_up_SAM_predict_with_prompt(sam=sam)

    for image in imported_images:  # args.target_list (list)
        if args.generate_mask:  # generating masks takes time
            masks = sam.generate(image)
        else:  # predicting masks with given prompts (e.g XY coordinates)
            pass
            # masks = masks, _, _ = sam.predict(prompt)

    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # _show_anns(masks)
    # plt.axis("off")
    # plt.show()
