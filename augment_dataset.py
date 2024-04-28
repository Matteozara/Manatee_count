from PIL import Image, ImageEnhance
import os
import random
import json
import h5py
import numpy as np
from PIL import Image
import shutil


def apply_brightness_contrast(image, brightness_factor, contrast_factor):
    # Adjust brightness and contrast
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    return image





training_data = "./data_path/train.json"
suffix = "dot"
folder_h5 = 'mydataset'
new_train_list = []

with open(training_data, "r") as outfile:
    train_list = json.load(outfile)


print(len(train_list))

c = 0
t = 0   #to modify only 1 image out of 4
for i in train_list:
    gt_path = i.replace('.jpg', suffix + '.h5').replace('images', folder_h5)
    img = Image.open(i)#.convert("RGB")
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file["density"])
    new_train_list.append(i)

    if target.sum() < 5:
        if t == 3:
            t = 0
            brightness_factor = random.uniform(0.5, 1.5)
            contrast_factor = random.uniform(0.5, 1.5)
            augmented_image = apply_brightness_contrast(img, brightness_factor, contrast_factor)
            augmented_image.save("./images/augmented_" + str(c) +".jpg")
            shutil.copy(gt_path, "./mydataset/augmented_" + str(c) + suffix + ".h5")

            new_train_list.append("./images/augmented_" + str(c) +".jpg")
            c += 1
        else:
            t += 1

with open("./data_path/train.json", 'w') as json_file:
    print("len new train: ", len(new_train_list))
    json.dump(new_train_list, json_file)

        



