# from segment_anything import SamPredictor, sam_model_registry

# sam = sam_model_registry["vit_b"](checkpoint="model_checkpoint/sam_vit_b_01ec64.pth")
# predictor = SamPredictor(sam)
# predictor.set_image("images/dog.jpeg")
# masks, _, _ = predictor.predict()

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torch

image = cv2.imread("images/dog.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
device = 'cuda'
sam = sam_model_registry["vit_b"](checkpoint="model_checkpoint/sam_vit_b_01ec64.pth")
sam.to('cuda' if torch.cuda.is_available() else 'cpu')
mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)
print(masks[0])

# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()