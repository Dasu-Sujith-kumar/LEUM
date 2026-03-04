import cv2
import os
import random
import numpy as np
import pandas as pd

orig_dir = "dataset/original"
out_dir = "dataset-balanced"

orig_out = os.path.join(out_dir, "original")
tamper_out = os.path.join(out_dir, "tampered")
mask_out = os.path.join(out_dir, "masks")

os.makedirs(orig_out, exist_ok=True)
os.makedirs(tamper_out, exist_ok=True)
os.makedirs(mask_out, exist_ok=True)

files = [f for f in os.listdir(orig_dir) if f.endswith(".png")]

random.shuffle(files)

copy_count = 5000
patch_count = 5000
intensity_count = 5000
original_count = 5000

metadata = []

# ------------------------
# 1. Untouched images
# ------------------------

for fname in files[:original_count]:

    img = cv2.imread(os.path.join(orig_dir, fname), 0)

    out_path = os.path.join(orig_out, fname)

    cv2.imwrite(out_path, img)

    metadata.append([fname, "original", 0])

# ------------------------
# 2. Copy Move
# ------------------------

start = original_count
end = start + copy_count

for fname in files[start:end]:

    img = cv2.imread(os.path.join(orig_dir, fname), 0)

    h, w = img.shape

    patch_w = random.randint(w//10, w//5)
    patch_h = random.randint(h//10, h//5)

    x = random.randint(0, w - patch_w)
    y = random.randint(0, h - patch_h)

    patch = img[y:y+patch_h, x:x+patch_w]

    x2 = random.randint(0, w - patch_w)
    y2 = random.randint(0, h - patch_h)

    tampered = img.copy()

    tampered[y2:y2+patch_h, x2:x2+patch_w] = patch

    mask = np.zeros((h,w), dtype=np.uint8)

    mask[y2:y2+patch_h, x2:x2+patch_w] = 255

    out_img = fname.replace(".png","_copy.png")
    out_mask = fname.replace(".png","_copy_mask.png")

    cv2.imwrite(os.path.join(tamper_out,out_img), tampered)
    cv2.imwrite(os.path.join(mask_out,out_mask), mask)

    metadata.append([out_img,"copy-move",1])

# ------------------------
# 3. Patch Insertion
# ------------------------

start = end
end = start + patch_count

for fname in files[start:end]:

    img1 = cv2.imread(os.path.join(orig_dir,fname),0)
    img2 = cv2.imread(os.path.join(orig_dir,random.choice(files)),0)

    h,w = img1.shape

    patch_w = random.randint(w//10,w//5)
    patch_h = random.randint(h//10,h//5)

    x = random.randint(0,w-patch_w)
    y = random.randint(0,h-patch_h)

    patch = img2[y:y+patch_h,x:x+patch_w]

    tampered = img1.copy()

    tampered[y:y+patch_h,x:x+patch_w] = patch

    mask = np.zeros((h,w),dtype=np.uint8)
    mask[y:y+patch_h,x:x+patch_w] = 255

    out_img = fname.replace(".png","_patch.png")
    out_mask = fname.replace(".png","_patch_mask.png")

    cv2.imwrite(os.path.join(tamper_out,out_img),tampered)
    cv2.imwrite(os.path.join(mask_out,out_mask),mask)

    metadata.append([out_img,"patch-insert",1])

# ------------------------
# 4. Intensity Modification
# ------------------------

start = end
end = start + intensity_count

for fname in files[start:end]:

    img = cv2.imread(os.path.join(orig_dir,fname),0)

    h,w = img.shape

    patch_w = random.randint(w//10,w//5)
    patch_h = random.randint(h//10,h//5)

    x = random.randint(0,w-patch_w)
    y = random.randint(0,h-patch_h)

    tampered = img.copy()

    roi = tampered[y:y+patch_h,x:x+patch_w]

    alpha = random.uniform(0.6,1.4)
    beta = random.randint(-40,40)

    tampered[y:y+patch_h,x:x+patch_w] = np.clip(alpha*roi+beta,0,255)

    mask = np.zeros((h,w),dtype=np.uint8)
    mask[y:y+patch_h,x:x+patch_w] = 255

    out_img = fname.replace(".png","_intensity.png")
    out_mask = fname.replace(".png","_intensity_mask.png")

    cv2.imwrite(os.path.join(tamper_out,out_img),tampered)
    cv2.imwrite(os.path.join(mask_out,out_mask),mask)

    metadata.append([out_img,"intensity-mod",1])

# ------------------------
# Save Metadata
# ------------------------

pd.DataFrame(
    metadata,
    columns=["image_id","tampering_type","label"]
).to_csv("metadata_balanced.csv",index=False)

print("Balanced dataset created")