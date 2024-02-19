# nested k-fold cross-validation study for training 2D and 3D AA images 
# train/validate (outerfold) and save final models
# last updated: 2024-02-15
# written by: Thomas Kierski & Kathlyne Bautista

# torch
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, SubsetRandomSampler
from torch import Tensor

# monai
import monai
from monai.data import Dataset, ImageDataset, DataLoader, ArrayDataset
from monai.transforms import AddChannel, Compose, RandRotate90, RandRotate, RandCoarseDropout, RandZoom, RandGaussianNoise, Resize, ScaleIntensity, NormalizeIntensity, EnsureType, ToDevice
from monai.networks.nets.efficientnet import get_efficientnet_image_size

# other packages
import sys
import os
import time
from os import makedirs, listdir
from os.path import join
import numpy as np
from scipy.io import loadmat
import random
from timeit import default_timer
from sklearn.model_selection import StratifiedKFold
import wandb
import argparse
import pickle

# Get path names from environment
data3d_path=os.getenv('DATA3D_PATH')
data2d_path=os.getenv('DATA2D_PATH')
data_dir=os.getenv('DATA_DIR')

print(data3d_path)
print(data2d_path)

# Init wandb
wandb.init(project=os.getenv('PROJECT'),name='FINAL_MODELS')

def save_model_config(path,config):
	with open(path,'wb') as file:
		pickle.dump(config,file,protocol=pickle.HIGHEST_PROTOCOL)
		
def load_model_config(path):
	with open(path,'rb') as file:
		return pickle.load(file)

def get_dirs(path):
	dirs = [x for x in listdir(path) if '.mat' in x]
	dirs = [x for x in dirs if not x.startswith('.')]
	dirs = [x for x in dirs if not x.startswith('_')]
	dirs = [join(path,x) for x in dirs]
	return dirs

def get_data(path):        
	dirs = get_dirs(path)
	out = [torch.from_numpy(loadmat(x,simplify_cells=True)['out']['imtor'].astype('float32')) for x in dirs]
	return out

def list_mean(lst):
	return sum(lst)/len(lst)

def val_acc_warning(best_val_acc):
	inp = input(f"Warning: Init val acc = {best_val_acc}, do you wish to proceed? (y/n) ")
	if inp not in ['y','Y']:
		sys.exit('exiting')
	
def check_best_score(path: str) -> float:
	# Path is directory where model checkpoints are saved. If no .dat file with scores is in the folder, we make one and init to 0.0
	data_path = os.path.join(path,'val_acc.dat')
	if os.path.isfile(data_path):
		with open(data_path,'r') as f:
			score = f.readline()
			if score == '': # in case there's an empty file for some reason
				score = 0.0
			print(f"file exists, score = {score}")
		return float(score)
	else:
		with open(data_path,'w') as f:
			score = 0.0
			print(f"no file, score = {score}")
			f.write(str(score))
		return score

def update_best_score(path: str, score: float) -> None:
	data_path = os.path.join(path,'val_acc.dat')
	with open(data_path,'w') as f:
		f.write(str(score))

def check_best_loss(path: str) -> float:
	# Path is directory where model checkpoints are saved. If no .dat file with scores is in the folder, we make one and init to 100000
	data_path = os.path.join(path,'val_loss.dat')
	if os.path.isfile(data_path):
		with open(data_path,'r') as f:
			loss = f.readline()
			if loss == '': # in case there's an empty file for some reason
				loss = 100000
			print(f"file exists, loss = {loss}")
		return float(loss)
	else:
		with open(data_path,'w') as f:
			loss = 100000 # init the loss to something very large 
			print(f"no file, loss = {loss}")
			f.write(str(loss))
		return loss

def update_best_loss(path: str, loss: float) -> None:
	data_path = os.path.join(path,'val_loss.dat')
	with open(data_path,'w') as f:
		f.write(str(loss))

class CustomImageDataset(Dataset):
	def __init__(self, control_dir, tumor_dir, transform=None, target_transform=None):
		self.control_imgs = self.get_dirs(control_dir)
		self.tumor_imgs = self.get_dirs(tumor_dir)
		self.img_paths = self.control_imgs + self.tumor_imgs
		self.img_labels = np.concatenate((np.zeros((len(self.control_imgs,))),np.ones((len(self.tumor_imgs,)))),axis=0)
		self.img_labels = torch.from_numpy(self.img_labels.astype('float32')) # change these two lines << ^^        
		self.transform = transform
		self.target_transform = target_transform

	def get_dirs(self,path):
		dirs = [x for x in listdir(path) if '.mat' in x]
		dirs = [x for x in dirs if not x.startswith('.')]
		dirs = [x for x in dirs if not x.startswith('_')]
		dirs = [join(path,x) for x in dirs]
		return dirs        
		
	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):      
		img_path = self.img_paths[idx]
		image = loadmat(img_path,simplify_cells=True)
		image = image['out']['imtor'].astype('float32')
		label = self.img_labels[idx]
		
		image = torch.from_numpy(image)
		
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label

def train():

	if torch.cuda.device_count() > 1:
		config_path = os.path.join(args.checkpoint_path,'trained_models',str(args.spatial_dims)+'d_models',args.network+'_mutliGPU')
	else:
		config_path = os.path.join(args.checkpoint_path,'trained_models',str(args.spatial_dims)+'d_models',args.network)
	
	outer_kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=int(2^26)) 
	# inner_kf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=int(2^24))  
	
	# Tracking the mean accuracy and loss metrics across the outer cross-validation loop
	train_acc_list = []
	train_loss_list =[]
	val_acc_list = []
	val_loss_list = []
	
	# Outer cross-validation loop
	for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(idx,torch.stack(labels).numpy())):
		print(f"outer fold: {outer_fold}")
		print(outer_test_idx)

		best_val_loss = 100.0
		best_val_acc = 0.0

		# Load best config
		fold_config_path = os.path.join(config_path,f'outer_fold_{outer_fold}/best_loss.config')
		config = load_model_config(fold_config_path)
		print(config)

		checkpoint_path = os.path.join(config_path,f'outer_fold_{outer_fold}')
		scheduler_step = 110
		scheduler_gamma = 0.2
		learning_rate = config['learning_rate']
		weight_decay = config['weight_decay']
		epochs = config['epochs']
		rotate_prob = config['rotate_prob']
		zoom_prob = config['zoom_prob']
		noise_prob = config['noise_prob']
		dropout_prob = config['dropout_prob']
		batch_size = config['batch_size']
		spatial_dims = config['spatial_dims']
		rotate_max = config['rotate_max']

		# Init the .dat files, don't care about returning anything here
		check_best_loss(checkpoint_path)
		check_best_score(checkpoint_path)

		# Setting up data augmentation pipeline
		if spatial_dims == 3:
			CD = RandCoarseDropout(holes=25,max_holes=50,spatial_size=5,max_spatial_size=20,fill_value=0.0,prob=dropout_prob)
			rotate = RandRotate(range_x=rotate_max, range_y=rotate_max, range_z=rotate_max, prob=rotate_prob, padding_mode="zeros")
		elif spatial_dims == 2:
			CD = RandCoarseDropout(holes=20,max_holes=35,spatial_size=5,max_spatial_size=20,fill_value=0.0,prob=dropout_prob)
			rotate = RandRotate(range_x=rotate_max, range_y=0, prob=rotate_prob, padding_mode="zeros")
		
		zoom = RandZoom(prob=zoom_prob)
		rgn = RandGaussianNoise(prob=noise_prob)
		si = ScaleIntensity(minv=0.0,maxv=1.0)
		
		train_transforms = Compose([ToDevice(device), CD, rgn, rotate, zoom, si, EnsureType()])
		val_transforms = Compose([ToDevice(device), EnsureType()])
		
		# Both datasets point to same data, only difference is the transform (no augmentations applied during inference)
		train_dataset = ArrayDataset(img=data,labels=labels,img_transform=train_transforms)
		val_dataset = ArrayDataset(img=data,labels=labels,img_transform=val_transforms)
		train_dl = DataLoader(train_dataset,batch_size=batch_size,sampler=outer_train_idx)
		val_dl = DataLoader(val_dataset,batch_size=batch_size,sampler=outer_test_idx)
		ntrain = len(outer_train_idx)
		ntest = len(outer_test_idx)

		# Set random seed to ensure that models are identically initialized on each run
		monai.utils.set_determinism(seed=2**31, additional_settings=None)

		torch.cuda.empty_cache()
		# Get model
		if config['network'] == 'DenseNet121':
			model = monai.networks.nets.DenseNet121(spatial_dims=spatial_dims, in_channels=1, out_channels=2).to(device)
		elif config['network'] == 'EfficientNetB0':
			model = monai.networks.nets.EfficientNetBN(model_name='efficientnet-b0',norm="instance",spatial_dims=spatial_dims, in_channels=1, num_classes=2).to(device)
		elif config['network'] == 'EfficientNetB1':
			model = monai.networks.nets.EfficientNetBN(model_name='efficientnet-b1',norm="instance",spatial_dims=spatial_dims, in_channels=1, num_classes=2).to(device)
		elif config['network'] == 'ResNet':
			model = monai.networks.nets.ResNet(block='basic', layers=[2,2,2,2], block_inplanes=[64,128,256,512], spatial_dims=spatial_dims, n_input_channels=1, num_classes=2).to(device)
		
		if torch.cuda.device_count() > 1:
			model = torch.nn.DataParallel(model)  
			
		optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		
		scheduler_step = 110
		scheduler_gamma = 0.2
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

		criterion = torch.nn.CrossEntropyLoss()
		scaler = torch.cuda.amp.GradScaler()
 
		pytorch_total_params = sum(p.numel() for p in model.parameters())
		print(f"\nModel size: {pytorch_total_params} parameters")
		print('\n')
   
		# Actual training loop below
		for ep in range(epochs):
			
			print(f"epoch: {ep}")
			
			# Training
			model.train()
			train_loss = 0.0    
			correct = 0
			t1 = default_timer()
			for _, (x,y) in enumerate(train_dl):         
				optimizer.zero_grad()
				x = x.to(device)
				y = y.to(device)
				with torch.cuda.amp.autocast():
					out = model(x)
					pred = torch.argmax(out,1)
					correct += (y==pred).sum()
					loss = criterion(out,y.type(torch.long))
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				train_loss += (loss.item()*x.shape[0]) # accounts for the batch size 

			train_loss/=ntrain
			train_acc = correct/ntrain
			train_acc = train_acc.item()
			scheduler.step()

			wandb.log({'Train loss':train_loss, 'Train accuracy':train_acc, 
				'outer_fold':outer_fold, 'Epoch':ep})

			# Validation
			model.eval()
			val_loss = 0.0
			correct = 0
			t1 = default_timer()
			with torch.no_grad():
				for _, (x,y) in enumerate(val_dl):                
					x = x.to(device)
					y = y.to(device)
					with torch.cuda.amp.autocast():
						out = model(x)          
						pred = torch.argmax(out,1)
						correct += (y==pred).sum()
						loss = criterion(out,y.type(torch.long))
					val_loss +=  (loss.item()*x.shape[0])

			val_loss/=ntest
			val_acc = correct/ntest
			val_acc = val_acc.item()

			wandb.log({'Test loss':val_loss, 'Test accuracy':val_acc,
				'outer_fold':outer_fold, 'Epoch':ep})

			# Keeping track of the best score for the fold rather than the finals 
			if val_loss < best_val_loss:
				print(f"\033[92mNew best model (loss) at epoch {ep+1}, val loss = {val_loss}\033[0m")     
				update_best_loss(checkpoint_path,val_loss)     
				PATH = os.path.join(checkpoint_path,'best_loss.pt')
				best_val_loss = val_loss
				torch.save({
					'epoch': ep,
					'outer_fold':outer_fold,
					'val_acc':val_acc,
					'train_acc':train_acc,
					'val_loss':val_loss,
					'train_loss':train_loss,
					'learning_rate': config['learning_rate'],
					'weight_decay': config['weight_decay'],
					'scheduler_gamma':scheduler_gamma,
					'scheduler_step':scheduler_step,
					'rotate_prob':config['rotate_prob'],
					'zoom_prob':config['zoom_prob'],
					'dropout_prob':config['dropout_prob'],
					'batch_size':config['batch_size'],
					'spatial_dims':config['spatial_dims'],
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					}, PATH)

			if val_acc > best_val_acc:
				print(f"\033[92mNew best model (accuracy) at epoch {ep+1}, val acc = {val_acc}\033[0m")
				update_best_score(checkpoint_path,val_acc)                
				PATH = os.path.join(checkpoint_path,'best_acc.pt')
				best_val_acc = val_acc
				torch.save({
					'epoch': ep,
					'outer_fold':outer_fold,
					'val_acc':val_acc,
					'train_acc':train_acc,
					'val_loss':val_loss,
					'train_loss':train_loss,
					'learning_rate': config['learning_rate'],
					'weight_decay': config['weight_decay'],
					'scheduler_gamma':scheduler_gamma,
					'scheduler_step':scheduler_step,
					'rotate_prob':config['rotate_prob'],
					'zoom_prob':config['zoom_prob'],
					'dropout_prob':config['dropout_prob'],
					'batch_size':config['batch_size'],
					'spatial_dims':config['spatial_dims'],
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					}, PATH)

		train_acc_list.append(train_acc)
		train_loss_list.append(train_loss)
		val_acc_list.append(best_val_acc)
		val_loss_list.append(best_val_loss)
		
	return train_acc_list, train_loss_list, val_acc_list, val_loss_list

if __name__ == '__main__':
	print(torch.__version__)
	torch.cuda.empty_cache()

	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8" 
	
	# Ensure deterministic behavior
	torch.use_deterministic_algorithms(True) 
	torch.backends.cudnn.benchmark = False # this is faster when true, but not deterministic 
	torch.backends.cudnn.deterministic = True

	my_seed = 123456
	random.seed(my_seed)
	np.random.seed(my_seed)
	torch.manual_seed(my_seed)
	torch.cuda.manual_seed_all(my_seed)
	monai.utils.set_determinism(seed=my_seed, additional_settings=None)

	parser = argparse.ArgumentParser()
	parser.add_argument('--no-save',help='Turns off saving checkpoints',action='store_true')
	parser.add_argument('-x','--checkpoint_path',help='Where to save the model checkpoint.',default=os.path.join(data_dir,'NESTED_KFOLD'))
	parser.add_argument('-v','--verbose',help='Increase output verbosity.',action='store_true')
	parser.add_argument('-p','--project',help='WandB project name',default='uncategorized')
	parser.add_argument('--batch_size',help='Default = 8',default=8,type=int)
	parser.add_argument('--dropout_prob',help='Default = 0.5',default=0.5,type=float)
	parser.add_argument('--epochs',help='Default = 10',default=10,type=int)
	parser.add_argument('--learning_rate',help='Default = 1e-4',default=1e-4,type=float)
	parser.add_argument('--network',help='Default = EfficientNetB0',default='EfficientNetB0',type=str)
	parser.add_argument('--noise_prob',help='Default = 0.5',default=0.5,type=float)
	parser.add_argument('--optimizer',help='Default = AdamW',default='AdamW',type=str)
	parser.add_argument('--rotate_max',help='Default = pi/12',default=0.2618,type=float)
	parser.add_argument('--rotate_prob',help='Default = 0.5',default=0.5,type=float)
	parser.add_argument('--spatial_dims',help='Default = 2',default=2,type=int,choices=[2,3])
	parser.add_argument('--weight_decay',help='Default = 1e-2',default=1e-2,type=float)
	parser.add_argument('--zoom_prob',help='Default = 0.5',default=0.5,type=float)

	args = parser.parse_args()
	
	img_size = get_efficientnet_image_size("efficientnet-b0") # 224x224x224 seems like good balance of resolution and memory requirement
	time_string = time.strftime("%m%d%y_%H:%M:%S",time.localtime())
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if args.spatial_dims == 3:
		data_path = data3d_path		
	elif args.spatial_dims == 2:
		data_path = data2d_path
	
	print(data_path)
	npzfile = np.load(data_path)
	X = torch.from_numpy(npzfile['x'].astype('float32'))
	Y = torch.from_numpy(npzfile['y'].astype('float32'))
	data = [x for x in X]
	labels = [y for y in Y]
	idx = [x for x in range(len(data))]

	if args.verbose:
		print(f"Device: {device}")
		print(f"Input size is {(img_size,)*args.spatial_dims}")
		print(f"Spatial dims = {args.spatial_dims}")
		print(f"Checkpoint path: {args.checkpoint_path}")
		print(f"No-save: {args.no_save}")

	train()
