import os
import glob
import cv2


dataset_dir = "/home/karlis/Desktop/dataset"

label_dir = "/home/karlis/Desktop/dataset/label"
image_dir = "/home/karlis/Desktop/dataset/image"

if not os.path.exists(label_dir):
    os.mkdir(label_dir)

if not os.path.exists(image_dir):
    os.mkdir(image_dir)

for img in glob.glob(os.path.join(dataset_dir, "*.jpg")):
    basename = os.path.basename(img)
    no_ext, _ = os.path.splitext(basename)

    mask_fn = f"{no_ext}.class-6.png"

    mask = cv2.imread(os.path.join(dataset_dir, mask_fn), cv2.IMREAD_GRAYSCALE)

    if cv2.countNonZero(mask) == 0:
        continue

    os.rename(os.path.join(dataset_dir, mask_fn), os.path.join(label_dir, mask_fn))
    os.rename(img, os.path.join(image_dir, basename))

    print("yes")
