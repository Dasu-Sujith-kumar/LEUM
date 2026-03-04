import cv2, os
import numpy as np

src = "cxr8_all_images"
dst = "cxr8_clean"
os.makedirs(dst, exist_ok=True)

removed = 0

for f in os.listdir(src):
    img_path = os.path.join(src, f)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        removed += 1
        continue

    if np.std(img) < 10:   # low contrast
        removed += 1
        continue

    cv2.imwrite(os.path.join(dst, f), img)

print("Total removed images:", removed)
