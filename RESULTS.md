# Classical vs YOLO results

Test split: 183 images (112 whole gates, 71 truncated gates).

| Metric | Classical | YOLO |
|---|---|---|
| Precision @ IoU 0.50 | 0.705 | 0.916 |
| Recall @ IoU 0.50 | 0.479 | 0.936 |
| mAP50 | 0.421 | 0.907 |
| Mean center error (normalized) | 0.048 | 0.009 |
| FPS (CPU) | 57 | 41 |

| Subset | Classical recall | YOLO recall | Classical mAP50 | YOLO mAP50 |
|---|---|---|---|---|
| Whole (112) | 0.797 | 0.942 | 0.717 | 0.912 |
| Truncated (71) | 0.169 | 0.930 | 0.128 | 0.913 |

## Analysis
Classical assumes both posts are in the frame, so it misses most truncated gates.
YOLO learns what a partial gate looks like and holds up on both subsets.
YOLO costs about 30% of the speed (41 vs 57 FPS), still real time.
I would deploy YOLO.

## Known limitations
Some source frames appear in more than one split, so YOLO's scores may be optimistic.
Not reported: orientation macro-F1, confusion matrix, side-by-side video.
