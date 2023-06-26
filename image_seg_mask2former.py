from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch


def predict_seg_img(image) -> str:
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to processor for postprocessing
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
    predicted_instance_map = result["segmentation"]

    return predicted_instance_map
