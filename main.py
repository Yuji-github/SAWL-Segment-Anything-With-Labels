from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import torch
import json
import numpy as np
from typing import List
import argparse
import os
from tqdm import tqdm
from PIL import Image

from display import output_images
from image_seg_mask2former import predict_seg_img
from cluster_images import ClusterImages


def parse_args() -> argparse:
    parser = argparse.ArgumentParser()

    # import images and target segmentation names
    parser.add_argument(
        "--image_folder_path",
        "-if",
        type=str,
        default="images",
        help="folder path for images",
        required=True,
    )
    parser.add_argument(
        "--target_list",
        "-tl",
        nargs="+",
        help="target list for segmentation",
        required=True,
    )

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
    parser.add_argument(
        "--kwargs_auto",
        "-ka",
        type=str,
        default=None,
        help="dict kwargs: json format required",
    )

    # prompts for SamPredictor
    parser.add_argument(
        "--prompts",
        "-pr",
        type=str,
        default=None,
        help="dict prompts: json format required",
    )

    # select generate masks or predict with prompts
    parser.add_argument(
        "--generate_mask",
        "-gm",
        type=eval,
        choices=[True, False],
        default=True,
        help="generating segmentation masks with SAM",
    )

    # Cluster
    parser.add_argument(
        "--model",
        "-mo",
        type=str,
        default="resnet-18",
        help="model for extracting features",
    )
    parser.add_argument(
        "--cluster",
        "-cl",
        type=str,
        default="hdbscan",
        help="cluster methods: dbscan or hdbscan",
    )

    # Threshold

    parser.add_argument(
        "--threshold",
        "-th",
        type=float,
        default=0.8,
        help="threshold for labeling",
    )

    # Display
    parser.add_argument(
        "--display",
        "-dis",
        type=eval,
        choices=[True, False],
        default=False,
        help="displaying masked image: True (display), False (not display)",
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


def select_index_from_cluster(cluster_number: int, labels: np.array) -> int:
    """Selecting a sample from each cluster labeling randomly and return index
    :param cluster_number:
    :param lables:
    :return int: index
    """
    group = np.where(labels == cluster_number)[0]
    return group[np.random.choice(group.shape[0])]


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
    print("#### Generating COCO Format")
    for image in tqdm(imported_images):  # args.target_list (list)
        masks = [{}]
        if args.generate_mask:  # generating masks takes time
            masks = sam.generate(image)
        else:  # predicting masks with given prompts (e.g, XY coordinates)
            masks, _, _ = sam.predict(
                args.prompts,
                point_coords=args.prompts["point_coords"] if "point_coords" in args.prompts else None,
                point_labels=args.prompts["point_labels"] if "point_labels" in args.prompts else None,
                box=args.prompts["box"] if "box" in args.prompts else None,
                mask_input=args.prompts["mask_input"] if "mask_input" in args.prompts else None,
                multimask_output=args.prompts["multimask_output"] if "multimask_output" in args.prompts else True,
                return_logits=args.prompts["return_logits"] if "return_logits" in args.prompts else False,
            )

        # Clustering
        sorted_area_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        cluster = ClusterImages(image=image, masks=sorted_area_masks, model=args.model, cluster=args.cluster)
        labels = cluster.create_image_cluster()

        # replace unique (-1) to positive discrete int
        num = labels.max() + 1
        for idx in range(len(labels)):
            if labels[idx] == -1:
                labels[idx] = num
                num += 1

        # Predicting sample objects from clustering
        for cluster_number in np.unique(labels):
            index = select_index_from_cluster(cluster_number, labels)
            pred_name, score = predict_seg_img(Image.fromarray(cluster.seg_images[index]))

            for idx in np.where(labels == cluster_number)[0]:
                if score >= args.threshold:
                    if pred_name in args.target_list:
                        sorted_area_masks[idx]["id"] = pred_name
                else:  # removing unnecessary masks
                    sorted_area_masks.pop(idx)

        if args.display:
            output_images(image, sorted_area_masks)
