from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from typing import Tuple


def predict_seg_img(image) -> Tuple[str, float]:
    """Predicting masked images with Mask2Former (800 MB)
    :param image:
    :return predict_name, score:
    """
    # downloading model from HuggingFace
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")

    # carrying the image to torch
    inputs = processor(images=image, return_tensors="pt")

    # giving the image to the model: return tensor values
    with torch.no_grad():
        outputs = model(**inputs)

    # predicting image with id values (int)
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    # converting id to word
    seg_info = result["segments_info"][0]
    segment_label_id = seg_info["label_id"]

    return model.config.id2label[segment_label_id], seg_info["score"]
