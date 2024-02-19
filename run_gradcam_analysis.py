# import packages
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import monai
from monai.data import Dataset, ImageDataset, DataLoader, ArrayDataset
from monai.networks.nets.efficientnet import get_efficientnet_image_size
from monai.transforms import AddChannel, Compose, RandRotate90, RandRotate, RandCoarseDropout, RandZoom, RandGaussianNoise, Resize, ScaleIntensity, NormalizeIntensity, EnsureType, ToDevice

import random
from sklearn.model_selection import StratifiedKFold

from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.visualize import GradCAM, CAM

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" 

# Turn interactive plotting off
plt.ioff()

torch.use_deterministic_algorithms(True) 
torch.backends.cudnn.benchmark = False # this is faster when true, but not deterministic 
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# define image size
img_size = get_efficientnet_image_size("efficientnet-b0")

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

# data directories
gradcam_dir = os.getenv('GRADCAM_DIR')
model_dir = os.getenv('MODEL_DIR')
data_path = os.getenv('DATA2D_PATH')

# define model
####### EfficientNetB0
model = monai.networks.nets.EfficientNetBN(model_name='efficientnet-b0',norm="instance",spatial_dims=2, in_channels=1, num_classes=2).to(device)
model_dir = model_dir+'/EfficientNetB0/'
targ_layer = '_conv_head'
    
####### EfficientNetB1
# model = monai.networks.nets.EfficientNetBN(model_name='efficientnet-b1',norm="instance",spatial_dims=2, in_channels=1, num_classes=2).to(device)
# model_dir = model_dir+'/EfficientNetB1/'
# targ_layer = '_conv_head'
    
####### DenseNet121
# model = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
# model_dir = model_dir+'/DenseNet121/'
# targ_layer = 'class_layers.relu' 

print("gradcam_dir = ",gradcam_dir)
print("data_path = ",data_path)
print("model_dir = ",model_dir)

npzfile = np.load(data_path)
X = torch.from_numpy(npzfile['x'].astype('float32'))
Y = torch.from_numpy(npzfile['y'].astype('float32'))
data = [x for x in X]
labels = [y for y in Y]
idx = [x for x in range(len(data))]

si = ScaleIntensity(minv=0.0,maxv=1.0)
val_transforms = Compose([ToDevice(device), EnsureType()])
val_dataset = ArrayDataset(img=data,labels=labels,img_transform=val_transforms)
kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=int(2^26))

# Tumors
for fold, (train_idx,test_idx) in enumerate(kf.split(idx,torch.stack(labels).numpy())):  
    torch.cuda.empty_cache()
    print("\033[93m"+"#"*100+"\033[0m")
    print(f"fold: {fold+1}")
    print(test_idx)
    ntrain = len(train_idx)
    ntest = len(test_idx)

    monai.utils.set_determinism(seed=2**31, additional_settings=None)
    val_dl = DataLoader(val_dataset,batch_size=1,sampler=test_idx)
    model_path = model_dir+'outer_fold_'+str(fold)+'/best_loss.pt'
    print(model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(next(model.parameters()).is_cuda) # returns a boolean

    cc = 0
    ii = 0
    ims = []
    cams = []
    with torch.no_grad():
        for _, (x,y) in enumerate(val_dl):  
            if y.item() == 1: # find tumors
                ims.append(x)
    print(len(ims))
    X = torch.cat(ims,axis=0)
    print(X.shape)
    
    X = X.to(device)
    model.eval()
    out = model(X)
    pred = torch.argmax(out,axis=1)

    model.train()

    cam = GradCAM(nn_module=model, target_layers=targ_layer)
    result = cam(x=X)

    result.shape

    fig = plt.figure(figsize=(15,15))
    for i in range(len(X)):
        if pred[i].cpu().item() == 1:
            save_dir = gradcam_dir+"/TP/"
        else:
            save_dir = gradcam_dir+"/FN/"

        fig = plt.imshow(np.squeeze(X[i,:,:,:].cpu().numpy()),cmap='gray')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(save_dir+"f"+str(fold)+"_t"+(str(pred[i].cpu().item())+"_"+str(i))+"img.png",dpi=300,
                    transparent=True,bbox_inches='tight', pad_inches = 0)
                    
        fig = plt.imshow(np.squeeze(result[i,:,:,:].cpu().numpy()),cmap='jet_r',alpha=0.35)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(save_dir+"f"+str(fold)+"_t"+(str(pred[i].cpu().item())+"_"+str(i))+"gc.png",dpi=300,
                    transparent=True,bbox_inches='tight', pad_inches = 0)
        

# Controls
for fold, (train_idx,test_idx) in enumerate(kf.split(idx,torch.stack(labels).numpy())):
       
    torch.cuda.empty_cache()
    print("\033[93m"+"#"*100+"\033[0m")
    print(f"fold: {fold+1}")
    print(test_idx)
    ntrain = len(train_idx)
    ntest = len(test_idx)

    monai.utils.set_determinism(seed=2**31, additional_settings=None)
    val_dl = DataLoader(val_dataset,batch_size=1,sampler=test_idx)
    model_path = model_dir+'outer_fold_'+str(fold)+'/best_loss.pt'
    print(model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(next(model.parameters()).is_cuda) # returns a boolean

    cc = 0
    ii = 0
    ims = []
    cams = []
    with torch.no_grad():
        for _, (x,y) in enumerate(val_dl):  
            if y.item() == 0: # find controls
                ims.append(x)
    print(len(ims))
    X = torch.cat(ims,axis=0)
    print(X.shape)
    
    X = X.to(device)
    model.eval()
    out = model(X)
    pred = torch.argmax(out,axis=1)

    model.train()

    cam = GradCAM(nn_module=model, target_layers=targ_layer)
    result = cam(x=X)

    result.shape
    
    fig = plt.figure(figsize=(15,15))

    for i in range(len(X)):
        if pred[i].cpu().item() == 0:
            save_dir = gradcam_dir+"/TN/"
        else:
            save_dir = gradcam_dir+"/FP/"

        fig = plt.imshow(np.squeeze(X[i,:,:,:].cpu().numpy()),cmap='gray')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(save_dir+"f"+str(fold)+"_t"+(str(pred[i].cpu().item())+"_"+str(i))+"img.png",dpi=300,
                    transparent=True,bbox_inches='tight', pad_inches = 0)
                    
        fig = plt.imshow(np.squeeze(result[i,:,:,:].cpu().numpy()),cmap='jet_r',alpha=0.35)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(save_dir+"f"+str(fold)+"_t"+(str(pred[i].cpu().item())+"_"+str(i))+"gc.png",dpi=300,
                    transparent=True,bbox_inches='tight', pad_inches = 0)