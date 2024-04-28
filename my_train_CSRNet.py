#!/usr/bin/env python
# coding: utf-8
import os
import json
import time
from torch.utils.data import DataLoader
import torch
import torch._utils_internal
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
#import torch.nn.functional as F
from CSRNet import CSRNet
#from smaller_CSRNet import CSRNet_smaller
#from UNet import UNet
from Utils.load_dataset import ClassDataset
from Utils.custom_round import custom_round

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DEFINE PARAMETERS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data = "./data_path/train.json"  # path to train json
valid_data = "./data_path/valid.json"
test_data = "./data_path/test.json"  # path to test json
suffix = "dot"
folder_h5 = 'mydataset'
task = "Finetune"  # name of folder to save the weights (inside weights folder)
pre = task + "checkpoint.pth.tar"  # path to the pretrained model
num_epochs = 1000  # Epoch
best_predic = 1e6  # Optimal accuracy
best_percent = 1e6
initail_learning_rate = 1e-4  # Initial learning rate
lr = 1e-4  # learning rate
batch_size = 1  #4  # batch_size
decay = 1e-4  # Learning rate decay
workers = 4  # Number of threads
seed = 42  # Random seeds
stand_by = 10
print_freq = 10  # Print queue

LR_TMAX = 10
LR_COSMIN = 1e-6

torch.manual_seed(seed)

# mkdir
if not os.path.isdir('./weights'):
    os.makedirs('./weights')
weight_save_dir = os.path.join('weights', task)
if not os.path.isdir(weight_save_dir):
    os.makedirs(weight_save_dir)

class my_loss(torch.nn.Module):
    
    def __init__(self):
        super(my_loss,self).__init__()
        #self.loss = nn.L1Loss()#torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=0.1)
        self.loss = nn.MSELoss()
    
    def forward(self, prediction, target):
        return self.loss(prediction, target)#torch.abs(target.sum() - prediction.sum())    #/ target.size(0) IN CASE OF BATCH

        

def test(test_loader, model):
    print('##   Test phase    ##')
    start = time.time()
    criterion = nn.L1Loss().to(device)
    model.eval()
    mae = 0
    correct = 0
    tot_error = 0
    tot = 0
    for i, (img, target) in enumerate(test_loader):
        img = img.to(device)
        #img = Variable(img)
        output = model(img)
        target2 = target.type(torch.FloatTensor).unsqueeze(1).to(device)
        #target = Variable(target)
        loss = criterion(output, target2)
        mae += loss.item()
        #mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).to(device))
        for k in range(0, img.size(0)):
            sum_predicted = output[k].data.sum()
            sum_gt = target[k].sum().type(torch.FloatTensor).to(device)
            if custom_round(sum_predicted) == sum_gt:
                correct += 1
            tot_error += abs(sum_gt - sum_predicted)
        tot += img.size(0)
            

    #mae = mae / tot
    print("MAE on Test: ", mae)
    print("Correct classified: ", correct, " on total classified: ", tot)
    print("Mean error on single image (per number of manatee): ", (tot_error/tot).item())
    print("Validation execution time (in sec): ", round(time.time() - start, 3))

    return mae


def validate(valid_loader, model, criterion):
    print('##   Validation phase    ##')
    start = time.time()
    #model.eval()
    criterion_mae = nn.L1Loss().to(device)
    mae = 0
    correct = 0
    tot_error = 0
    tot = 0
    for i, (img, target) in enumerate(valid_loader):
        #img = Variable(img)
        img = img.to(device)
        output = model(img)
        target2 = target.type(torch.FloatTensor).unsqueeze(1).to(device)
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
        #tot_error += abs(sum_gt - sum_predicted)
        '''for k in range(0, img.size(0)):
            sum_predicted = output[k].data.sum()
            sum_gt = target[k].sum().type(torch.FloatTensor).to(device)
            print("sum predict: ", sum_predicted.item())
            print("sum target: ", sum_gt.item())
            if custom_round(sum_predicted) == sum_gt:
                correct += 1
            tot_error += abs(sum_gt - sum_predicted)'''
        tot += img.size(0)
        #print("tot: ", tot)

    #mae = mae / tot
    print("MAE on Validation (Pytorch): ", mae)
    print("Correct classified: ", correct, " on total classified: ", tot)
    print("MAE on single image (per number of manatee): ", (tot_error/tot))
    print("Validation execution time (in sec): ", round(time.time() - start, 3))

    percent = tot_error/tot #round((correct * 100) / tot, 2)

    return mae, percent


def train(train_loader, model, criterion, optimizer):
    print('Learning rate: ', optimizer.param_groups[0]['lr'] * 10000)
    start = time.time()
    tot_loss = 0
    model.train()
    for i, (img, target) in enumerate(train_loader):
        img = img.to(device)

        #img = Variable(img)
        output = model(img)
        target = target.type(torch.FloatTensor).unsqueeze(1).to(device)
        #target = Variable(target)
        loss = criterion(output, target)
        tot_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # torch.to(device).empty_cache()
    print("Epoch finished finished!")
    print("time: ", round(time.time() - start, 3))
    print("total loss of this epoch: ", tot_loss.item())

    return model
    

def save_checkpoint(state, task_id, best_percent, model):
    #mae = round(mae * 1e06, 3)
    ckpt_path = os.path.join(weight_save_dir, task_id + '_epoch_' + str(state['epoch']) + '_mae_' + str(best_percent) + '.pth')
    torch.save(model.state_dict(), ckpt_path)
    #torch.save(state, ckpt_path)


#OPEN DATA
transform = transforms.ToTensor() #transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

with open(training_data, "r") as outfile:
    train_list = json.load(outfile)
with open(valid_data, "r") as outfile:
    val_list = json.load(outfile)
with open(test_data, "r") as outfile:
    test_data = json.load(outfile)

lista_train = ClassDataset(train_list, transform, suffix, folder_h5)
train_loader = DataLoader(lista_train, batch_size=batch_size, shuffle=True, num_workers=workers)

lista_valid = ClassDataset(val_list, transform, suffix, folder_h5)
valid_loader = DataLoader(lista_valid, batch_size=batch_size, shuffle=False, num_workers=workers)

lista_test = ClassDataset(test_data, transform, suffix, folder_h5)
test_loader = DataLoader(lista_test, batch_size=batch_size, shuffle=False, num_workers=workers)


'''#DEFINE MODEL
model = CSRNet()
model = model.to(device)

checkpoint = torch.load("pretrained/African_epoch_31_mae_4.30486222743988.pth.tar") #pretrained/Human_epoch_24_mae_360.0378328928581.pth.tar #pretrained/Malaria_epoch_10_mae_0.45892569893284846.pth.tar
model.load_state_dict(checkpoint['state_dict'])'''
'''pretrained_model_path = 'pretrained/prva.pth.tar'
pretrained_model = torch.load(pretrained_model_path)
pretrained_state_dict = pretrained_model['state_dict']
'''
print("CSRNet on dataset: africa")
model = CSRNet()
model = model.to(device)
model.load_state_dict(torch.load('pretrained/CSRNet_AfricaWildlife_final.pth'))
#model.eval()
#print("dict: ", pretrained_model)
#model.load_state_dict(pretrained_state_dict)
model.eval()

#MODEL PARAMETERS
criterion = my_loss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), cfg["lr"], weight_decay=cfg["decay"])
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=LR_TMAX, eta_min=LR_COSMIN)

#TRAIN
for epoch in range(0, num_epochs):
    print("-------------- EPOCH ", epoch+1, " --------------")
    model = train(train_loader, model, criterion, optimizer)
    prec1, percent = validate(valid_loader, model, criterion)

    '''if prec1 < best_predic:
        print("mae: ", prec1, " best_predict: ", best_predic)
        best_predic = prec1
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_predic,
            'optimizer': optimizer.state_dict()
        }, task, best_predic)'''
    if best_percent > percent:
        print("percent: ", percent, " best_percent: ", best_percent)
        best_percent = percent
        best_predic = prec1
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_predic,
            'optimizer': optimizer.state_dict()
        }, task, best_percent, model)

    print('### best percentage: ', best_percent)

    scheduler.step()

print("###  FINAL TEST ON TEST SET   ###")
final_mae = test(test_loader, model)
