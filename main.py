from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import argparse
import os
from tqdm import tqdm

from image_seg_mask2transformer import predict_seg_img


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

    # kwargs for SamAutomaticMaskGenerator
    parser.add_argument("--kwargs_auto", "-ka", type=str, default=None, help="dict kwargs: json format required")

    # prompts for SamPredictor
    parser.add_argument("--prompts", "-pr", type=str, default=None, help="dict prompts: json format required")

    # select generate masks or predict with prompts
    parser.add_argument(
        "--generate_mask",
        "-gm",
        type=eval,
        choices=[True, False],
        default=True,
        help="generating segmentation masks with SAM",
    )

    return parser.parse_args()


def _import_images(images: list) -> List[np.array]:
    """Importing images to cv2, convert the images to RGB
    SAM requires HWC uint8 format

    :param images:
    :return List[cv2]:
        RGB images in the list
    """
    if not images:
        return []

    imported_images = []
    for image in tqdm(images):
        imported_images.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    return imported_images


def _set_up_SAM_mask_generator(sam: sam_model_registry = None, params: dict = {}) -> SamAutomaticMaskGenerator:
    """Setting SAM mask generator
    Assume users do not give input prompts such as point is in (X,Y) in pixels

    :param sam:
        SAM model
    :param params:
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
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=params["points_per_side"] if "points_per_side" in params else 32,
        points_per_batch=params["points_per_batch"] if "points_per_batch" in params else 64,
        pred_iou_thresh=params["pred_iou_thresh"] if "pred_iou_thresh" in params else 0.88,
        stability_score_thresh=params["stability_score_thresh"] if "stability_score_thresh" in params else 0.95,
        stability_score_offset=params["stability_score_offset"] if "stability_score_offset" in params else 1.0,
        box_nms_thresh=params["box_nms_thresh"] if "box_nms_thresh" in params else 0.7,
        crop_n_layers=params["crop_n_layers"] if "crop_n_layers" in params else 0,
        crop_nms_thresh=params["crop_nms_thresh"] if "crop_nms_thresh" in params else 0.7,
        crop_n_points_downscale_factor=params["crop_n_points_downscale_factor"]
        if "crop_n_points_downscale_factor" in params
        else 1,
        point_grids=params["point_grids"] if "crop_n_layers" in params else None,
        min_mask_region_area=params["min_mask_region_area"] if "min_mask_region_area" in params else 0,
    )


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


if __name__ == "__main__":
    args = parse_args()

    # Reading images
    images = [args.image_folder_path + "/" + file for file in os.listdir(args.image_folder_path)]
    print(f"#### Reading Images at {args.image_folder_path}")
    imported_images = _import_images(images)

    # SAM registry
    # model_type: recommend vit_h (2.4 GB) to get more segmentation masks
    # checkpoint: this model must match with the model_types
    # if GPU available, put SAM on GPU otherwise CPU (CPU takes time)
    print(f"#### Reading Check Point Model at {args.checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")

    # Building model
    print("#### Building Model")
    if args.generate_mask:  # generate masks (default)
        params = {}
        if args.kwargs_auto:
            with open(args.kwargs_auto, "r") as f:
                params = json.loads(f.read())
        sam = _set_up_SAM_mask_generator(sam=sam, params=params)
    else:  # predict masks
        sam = _set_up_SAM_predict_with_prompt(sam=sam)

    # generating COCO formats annotation data with given target
    # for image in imported_images:  # args.target_list (list)
    #     if args.generate_mask:  # generating masks takes time
    #         masks = sam.generate(image)
    #     else:  # predicting masks with given prompts (e.g, XY coordinates)
    #         pass
    #         # masks = masks, _, _ = sam.predict(prompt)

    image = imported_images[0]
    masks = sam.generate(image)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    _mapping_masks(masks)
    plt.axis("off")
    plt.show()
