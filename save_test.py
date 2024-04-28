#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import time
from torch.utils.data import DataLoader
import torch
import torch._utils_internal
import torch.nn as nn
from torchvision import transforms
#from torch.optim.lr_scheduler import CosineAnnealingLR
#import torch.nn.functional as F
from CSRNet import CSRNet
from Utils.load_dataset import ClassDataset
from Utils.custom_round import custom_round

def save_depth_map(depth_tensor, output_path):
    # Create a plot
    depth_tensor = depth_tensor.cpu().detach().numpy()
    depth_tensor = depth_tensor.squeeze()
    plt.imshow(depth_tensor) #, extent=(0, width, 0, height), origin='lower', cmap='viridis')
    '''plt.colorbar(label='Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()'''
    plt.savefig(output_path)
    '''# Normalize the depth map to values between 0 and 1
    depth_normalized = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())

    # Convert PyTorch tensor to PIL Image
    depth_image = transforms.ToPILImage()(depth_normalized)

    # Save the depth map as a PNG file
    depth_image.save(output_path)'''



os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DEFINE PARAMETERS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = "./data_path/test.json"  # path to test json
suffix = "dot"
folder_h5 = 'mydataset'
batch_size = 1  #4  # batch_size
workers = 4  # Number of threads
seed = 42  # Random seeds

torch.manual_seed(seed)


def test(test_loader, model):
    start = time.time()
    #model.eval()
    criterion_mae = nn.L1Loss().to(device)
    mae = 0
    correct = 0
    tot_error = 0
    tot = 0
    for i, (img, target) in enumerate(test_loader):
        #img = Variable(img)
        img = img.to(device)
        output = model(img)
        target2 = target.type(torch.FloatTensor).unsqueeze(1).to(device)
        save_depth_map(output, 'tests/depth_map ' + str(i) + '_predicted.png')
        save_depth_map(target, 'tests/depth_map ' + str(i) + '_original.png')
        #target = Variable(target)
        loss = criterion_mae(output, target2)
        mae += loss.item()
        
        sum_predicted = output.data.sum()
        sum_gt = target.sum().type(torch.FloatTensor).to(device)
        #print("sum predict: ", sum_predicted.item())
        #print("sum target: ", sum_gt.item())
        tot_error += torch.abs(output.sum() - target.sum()).item()
        if custom_round(sum_predicted) == sum_gt:
            correct += 1
        tot += img.size(0)
        #print("tot: ", tot)

    #mae = mae / tot
    print("MAE on Test (Pytorch): ", mae)
    print("Correct classified: ", correct, " on total classified: ", tot)
    print("MAE on single image (per number of manatee): ", (tot_error/tot))
    print("Test execution time (in sec): ", round(time.time() - start, 3))

    percent = round((correct * 100) / tot, 2)
    print("Percentage correct classified: ", percent)

    


#OPEN DATA
transform = transforms.ToTensor() #transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

with open(test_data, "r") as outfile:
    test_data = json.load(outfile)

lista_test = ClassDataset(test_data, transform, suffix, folder_h5)
test_loader = DataLoader(lista_test, batch_size=batch_size, shuffle=False, num_workers=workers)

# mkdir
if not os.path.isdir("./tests"):
    os.makedirs("./tests")


#DEFINE MODEL
print("CSRNet")
model = CSRNet()
model = model.to(device)
model.load_state_dict(torch.load('model_to_test/best.pth')) #smallerCSRNet


print("###  TEST ON TEST SET   ###")
test(test_loader, model)
