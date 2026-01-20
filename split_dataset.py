import os
import random
import shutil

BASE_DIR = "dataset"
TRAIN_RATIO = 0.8
CLASSES = ["healthy", "diseased"]

def collect_images(base_path):
    images = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, f))
    return images

for cls in CLASSES:
    src_base = os.path.join(BASE_DIR, "test", cls)
    train_dir = os.path.join(BASE_DIR, "train", cls)
    test_dir = os.path.join(BASE_DIR, "test", cls)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = collect_images(src_base)
    random.shuffle(images)

    split = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split]
    test_imgs = images[split:]

    # Move train images
    for img_path in train_imgs:
        fname = os.path.basename(img_path)
        shutil.move(img_path, os.path.join(train_dir, fname))

    # Move test images back to flat test folder
    for img_path in test_imgs:
        fname = os.path.basename(img_path)
        shutil.move(img_path, os.path.join(test_dir, fname))

    print(f"{cls}: {len(train_imgs)} train, {len(test_imgs)} test")
