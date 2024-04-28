import cv2
import numpy as np
import json
import os  # read files, operating systems
import cv2  # opencv in python
#import matplotlib.pyplot as plt
import numpy as np
'''import scipy
import scipy.spatial
import scipy.ndimage'''
import h5py
#import sys
from tqdm import tqdm

'''def create_bounding_box(width, height):
    center_x = (width - 1) / 2  # Calculate the center x-coordinate
    center_y = (height - 1) / 2  # Calculate the center y-coordinate

    bounding_box = np.zeros((height, width))  # Initialize a matrix of zeros

    for i in range(height):
        for j in range(width):
            distance_x = abs(j - center_x)
            distance_y = abs(i - center_y)
            bounding_box[i, j] = np.exp(-0.5 * (distance_x + distance_y))   #np.exp(-0.1 * (distance_x**2 + distance_y**2))
    
    bounding_box /= np.sum(bounding_box)  # Normalize to ensure sum is 1

    return bounding_box'''

def create_bounding_box(width, height):
    mu = [width/2, height/2]  # Mean
    sigma = [width/4, height/4]  # Standard deviation
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    x, y = np.meshgrid(x, y)

    gaussian = np.exp(-((x - mu[0])**2 / (2 * sigma[0]**2) + (y - mu[1])**2 / (2 * sigma[1]**2)))
    gaussian /= np.sum(gaussian)  # Normalize so it sums to 1
    gaussian *= 1 # Multiply by 100 to scale

    return gaussian

def generate_density_map(image_size, box_sizes): #box_sizes=[[sx, ex, sy, ey], ...]
    density_map = np.zeros((image_size[0], image_size[1]))

    for i in range(0, len(box_sizes)):
        '''center_x = np.random.randint(box_size[i], image_size[1] - box_size[i])
        center_y = np.random.randint(box_size, image_size[0] - box_size)

        # Create a bounding box
        top_left = (center_x - box_size, center_y - box_size)
        bottom_right = (center_x + box_size, center_y + box_size)'''
        if box_sizes[i][0] > box_sizes[i][1]: #sx > ex
            #swipe
            temp = box_sizes[i][0]
            box_sizes[i][0] = box_sizes[i][1]
            box_sizes[i][1] = temp
        
        '''if box_sizes[i][1] > image_size[1]:
            box_sizes[i][1] = image_size[1]
        if box_sizes[i][0] < 0:
            box_sizes[i][0] = 0'''

        width = box_sizes[i][1] - box_sizes[i][0]

        if box_sizes[i][2] > box_sizes[i][3]: #sy > ey
            #swipe
            temp = box_sizes[i][2]
            box_sizes[i][2] = box_sizes[i][3]
            box_sizes[i][3] = temp

        '''if box_sizes[i][3] > image_size[0]:
            box_sizes[i][3] = image_size[0]
        if box_sizes[i][2] < 0:
            box_sizes[i][2] = 0'''
        
        height = box_sizes[i][3] - box_sizes[i][2]

        bounding_box = create_bounding_box(width, height) #because the image is represented in this order (flipped)
        '''print("width and height: ", width, height)
        print("bunding box dimensions: ", bounding_box.shape)
        #print("corners: sx,ex,sy,ey ", box_sizes[i][0], box_sizes[i][1], box_sizes[i][2], box_sizes[i][3])
        print("box: ", box_sizes[i][2], box_sizes[i][3], box_sizes[i][0], box_sizes[i][1])
        print("density map shape: ", density_map[box_sizes[i][2]:box_sizes[i][3], box_sizes[i][0]:box_sizes[i][1]].shape)
        #print("bounding box shape: ", density_map[box_sizes[i][2]:box_sizes[i][3], box_sizes[i][0]:box_sizes[i][1]])'''

        density_map[box_sizes[i][2]:box_sizes[i][3], box_sizes[i][0]:box_sizes[i][1]] = bounding_box + density_map[box_sizes[i][2]:box_sizes[i][3], box_sizes[i][0]:box_sizes[i][1]]

    return density_map

img_root = "images/"
label_root = "json/"

#np.set_printoptions(threshold=sys.maxsize)


# get all image of the dataset
img_paths = []
for root, dirs, files in os.walk(img_root):
    for img_path in files:
        img_paths.append(os.path.join(root, img_path))

# create dataset
for img_path in tqdm(img_paths):
    # get the path of the GT
    gt_path = img_path.replace("images", "json")[:-3] + "json"
    if not os.path.isfile(gt_path):
        continue
    gt = []
    # read gt line by line
    with open(gt_path, "r") as json_file:
        label_data = json.load(json_file)
        lines = label_data["boxes"]
        num_manatee = label_data["human_num"]
        img_id = label_data["img_id"]
        for line in lines:
            gt.append([int(line["sx"]), int(line["ex"]), int(line["sy"]), int(line["ey"])])
    # load the image
    image = cv2.imread(img_path)
    # generate the density map
    positions = generate_density_map(image.shape, np.array(gt))
    count = 0
    for i in positions:
        for j in i:
            if j != 0.0:
                count += 1
    
    
    # assert(num_manatee == np.round(density_map.sum())), f"{img_path}, {num_manatee}, {density_map.sum()}"
    if not num_manatee == np.round(positions.sum()):
        print("Error on numbers generated")
        print(f"{img_path}, {num_manatee}, {positions.sum()}")
        continue

    # mkdir
    if not os.path.isdir('./mydataset'):
        os.makedirs('./mydataset')
        
    # save the density map
    with h5py.File((img_path[:-4] + 'dot.h5').replace('images', 'mydataset'), 'w') as hf:
        hf['density'] = positions
