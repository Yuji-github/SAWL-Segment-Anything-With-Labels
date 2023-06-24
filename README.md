# Mask2Transformer vs SAM

The below figures show the difference between Mask2Transformer and SAM.
- Mask2Transformer does not capture the individual objects, and it returns "grass", "tree", and "house".
- SAM captures each object well, but SAM does not know name of the objects.

<p align="center">
  <img src="readme_images/compare.png" alt="pre-commit" width="600" height="200">
</p>

# SAM Checkpoint Model Differences:
The below figures show that the differences of the model performance.
- Vit-B (400 MB) captures the small parts of objects such as nose.
- Vit-L (1.2 GB) captures larger parts of objects such as ears, but not capture the entire dog yet.
- Vit-H (2.4 GB) captures any sizes of objects including the entire dog.

We recommend to use Vit-H.

<p align="center">
  <img src="readme_images/models.png" alt="pre-commit" width="600" height="200">
</p>
