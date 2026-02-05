# Custom Data Guide

This project expects grayscale images that are resized to a fixed resolution
(default: 64x64) and flattened to vectors.

## Use Your Own Images

1. Place images in a folder, e.g. `data/custom_images/`.
2. Update the loader call in [physically_inspired_pattern_matching.py](physically_inspired_pattern_matching.py)
   to point at your folder.

Example:

```python
images, filenames = load_kitti_images(
    dataset_path="data/custom_images",
    resolution=(64, 64),
    n_samples=20,
)
```

## Notes

- Images are converted to grayscale and normalized to [0, 1].
- If you need a different resolution, change `resolution` consistently in the
  script and model configuration.
- For large datasets, lower `n_samples` during debugging.
