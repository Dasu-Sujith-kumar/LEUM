import os, random, shutil

src = "cxr8_clean"
dst = "dataset/original"
os.makedirs(dst, exist_ok=True)

files = random.sample(os.listdir(src), 20000)

for i, f in enumerate(files):
    shutil.copy(
        os.path.join(src, f),
        os.path.join(dst, f"cxr_{i:04d}.png")
    )
