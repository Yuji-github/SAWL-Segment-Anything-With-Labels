# from segment_anything import SamPredictor, sam_model_registry

# sam = sam_model_registry["vit_b"](checkpoint="model_checkpoint/sam_vit_b_01ec64.pth")
# predictor = SamPredictor(sam)
# predictor.set_image("images/dog.jpeg")
# masks, _, _ = predictor.predict()


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torch
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


image = cv2.imread("images/dog.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
device = "cuda"
sam = sam_model_registry["vit_b"](checkpoint="model_checkpoint/sam_vit_b_01ec64.pth")
sam.to("cuda" if torch.cuda.is_available() else "cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)
print(masks[0])
with open("t.json", "w") as t:
    json.dump(masks, t, cls=NumpyEncoder)


# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()
