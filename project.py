#!/usr/bin/env python
# coding: utf-8

# In[200]:


import torch
import glob 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,Dataset
# from google.colab import drive
import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from skimage.transform import resize
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score


# In[17]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[18]:


drive.mount('/content/gdrive')


# In[ ]:


get_ipython().system("unzip -q '/content/gdrive/MyDrive/Data/archive.zip' -d '/content/gdrive/MyDrive/data'")


# In[126]:


class cardio(Dataset):
  def __init__(self, root, transform, mood):
    """ plug in the path """
    self.root = root
    self.transform = transform
    dic = {}
    self.images = list()
    self.labels = list()
    self.mood = mood

    if self.mood == 'train':
      self.paths = glob.glob('Train/**/*.nii')
      files = [nib.load(path) for path in self.paths]

      for f in files:
        name = f.get_filename().split('/')[-1]
        if name in dic:
          dic[name].append(f.get_fdata()) 
        else:
          dic[name] = [f.get_fdata()] 
      
      for (label3D,image3D) in dic.values():
        for i in range(image3D.shape[-1]):
          if (label3D[:,:,i] > 0).any():
            self.images.append(image3D[:,:,i].reshape(1, 320, 320))
            self.labels.append(label3D[:,:,i].reshape(320, 320))
    else:
      self.paths = glob.glob('imagesTs/*.nii')
      files = [nib.load(path) for path in self.paths]
      for fl in files:
        image3D = fl.get_fdata()
        for i in range(image3D.shape[-1]):
          resized = resize(image3D[:,:,i], (320, 320))
          self.images.append(np.expand_dims(resized, 0))
          
      
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    if self.mood == 'train':
      image = self.images[index]
      label = self.labels[index]

      if (self.transform):
          image = self.transform(image)
          label = self.transform(label)
      return image, label
    else:
      image = self.images[index]
      return image


# In[23]:


# paths = glob.glob('gdrive/MyDrive/data/imagesTs/*.nii')
# files = [nib.load(path) for path in paths]
# len(files)
test_data = cardio('', transform=None, mood='test')
# images = []
# paths = glob.glob('imagesTs/*.nii')
# files = [nib.load(path) for path in paths]
# for fl in files:
#   image3D = fl.get_fdata()
#   print(image3D.shape)
#   # image3D[:,:,0].resize(320,320)
#   # for i in range(image3D.shape[-1]):
#   #   images.append(np.expand_dims(image3D[:,:,i], 0))


# In[26]:


plt.imshow(test_data[300].squeeze(0))


# In[144]:


def vis(dt):
  plt.figure(figsize=(4,8))
  for d in dt:
    plt.imshow(d, cmap='gray')
    plt.show()


# In[127]:


data = cardio('', transform=None, mood='train')

# visulalise(y)
# visulalise(x)


# In[78]:


len(data)


# In[ ]:


len(data)


# In[ ]:


len([item.sum() for item in data.labels if item.sum() > 0]) / len(data)


# # New Section

# In[128]:


class Unet(nn.Module):
  def __init__(self, num_classes=1000):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5, stride=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=5, stride=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=2, stride=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=2, stride=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 128, kernel_size=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, kernel_size=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, kernel_size=5),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 1, kernel_size=5),
        nn.Sigmoid(), # Check
    )


    # self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((6, 6))
    # self.flatten = nn.Flatten(start_dim=1)
    # self.classifier = nn.Sequential(
    #     nn.Linear(224*224*512, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, num_classes),
    #     nn.ReLU(inplace=True),
    # )

  def forward(self, x):
    y = nn.Sequential(self.encoder, self.decoder)(x)
    # y = self.encoder(x)
    # x = self.adaptive_avg_pool2d(x)
    # x = self.flatten(x)
    # x = self.classifier(x)
    return y


# In[195]:


model = smp.Unet(
        encoder_name= 'efficientnet-b4',
        encoder_depth=5,
        encoder_weights= 'imagenet',
        decoder_use_batchnorm=True,
        decoder_channels=[512, 256, 128, 64, 32],
        # decoder_attention_type= null,
        in_channels= 1,
        )


# In[199]:


# model = Unet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, )


# In[197]:



train_data, val_data = torch.utils.data.random_split(data, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))


# In[198]:


train_dl = DataLoader(train_data, batch_size= 32, shuffle=True, num_workers=4)
valid_dl = DataLoader(val_data, batch_size= 32, num_workers=4) #check , increase


# In[ ]:


# for data in train_dl:
#   print(data)
#   break

# (labels > 0).sum(), outputs.numel()


# In[172]:


def compute_accuracy(labels, predicted):
  acc = list()
  for label, pred in zip(labels, predicted):
    acc.append((label & pred).sum() / (label | pred).sum())
  return acc


# In[204]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    model = torch.nn.DataParallel(model)


# In[202]:


def evaluate(data_loader, mode):
    model.eval()
    total_loss = 0
    total = 0
    acc_list = list()

    for (images, labels) in data_loader:
        images = images.to(device).float()
        labels = labels.to(device).float()
        with torch.no_grad():
            outputs = model(images)
            labels = labels.reshape(-1,1,320,320)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            outputs = outputs > 0.5
            labels = labels.type(torch.BoolTensor)
            acc_list += jaccard_score(labels.to(device), outputs.to(device), average='macro')
            # acc_list += compute_accuracy(labels.to(device), outputs.to(device))
    accuracy = sum(acc_list) / len(acc_list)
    losss = total_loss / total
    print(f'{mode} Loss({losss:6.4f}) Accuracy ({accuracy:6.4f})')
    return accuracy


# In[211]:


model = model.to(device)
epochs = 200
for epoch in range(epochs):
  model.train()
  acc_list = list()
  total = 0
  total_correct = 0
  total_loss = 0
  best_model = 0
  for i, (images, labels) in enumerate(train_dl):
    images = images.to(device).float()
    labels = labels.to(device).float()
    # print(images.reshape(320,320,1).shape)
    optimizer.zero_grad()
    outputs = model(images)
    labels = labels.reshape(-1,1,320,320)
    # compute_accuracy 
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total += labels.numel()
    # outputs[outputs >= 0.5] = 1
    # outputs[outputs < 0.5] = 0
    outputs = outputs > 0.5
    labels = labels.type(torch.BoolTensor)
    acc_list += jaccard_score(labels.to(device), outputs.to(device), average='macro')
    # acc_list += compute_accuracy(labels.to(device), outputs.to(device))
    total_loss += loss.item() * images.size(0) # Check: how to compute accuracy for such problems
  losss = total_loss/total
  accuracy = sum(acc_list)/ len(acc_list) 
  print(f'Train epoch {epoch}: Loss({losss:6.8f}) Accuracy ({accuracy:6.8f})')
  val_acc = evaluate(valid_dl, 'Validation')
  if epoch > 30 and best_model < val_acc:
    torch.save(model, f'smp_model_{epoch}.pt')
    best_model = val_acc


# In[210]:


CUDA_LAUNCH_BLOCKING=1


# In[ ]:


total_loss


# In[ ]:


torch.save(model.state_dict(), '/content/gdrive/MyDrive/Data/modewl-dict.pt')


# In[ ]:





# In[167]:


mdl = torch.load('model.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    mdl = torch.nn.DataParallel(mdl)


# In[34]:


mdl = torch.load('model.pt')


# In[188]:


# def evaluate(model, loader, device, criterion, mode='validation'):
mdl.eval()
total_correct = 0
total_loss = 0
total = 0
acc_list = list()
for i, (images, labels) in enumerate(valid_dl):
  images = images.to(device).float()
  labels = labels.to(device).float()
  with torch.no_grad():
    outputs = mdl(images)
    labels = labels.reshape(-1,1,320,320)
    loss = criterion(outputs, labels)
    total_loss += loss.item() * images.size(0)
    total += images.size(0)
    # _, predictions = outputs.max(1)
    # total_correct += (labels == predictions).sum()
    outputs = outputs > 0.5
    labels = labels.type(torch.BoolTensor)
    acc_list += compute_accuracy(labels.to(device), outputs.to(device))
accuracy = sum(acc_list) / len(acc_list)

losss = total_loss / total
# accuracy = total_correct / total
print(f'Validation Loss({losss:6.4f}) Accuracy ({accuracy:6.4f})')


# In[141]:


i = 0

for i, (images, labels) in enumerate(valid_dl):
  images = images.to(device).float()
  labels = labels.to(device).float()
  labels = labels.reshape(-1,1,320,320)
  with torch.no_grad():
    outputs = mdl(images)
    outputs = outputs > 0.5
  if(i>3):
    break


# In[142]:


labels.shape


# In[145]:


vis([images[5][0].cpu().detach().numpy(),labels[5][0].cpu().detach().numpy(),outputs[5][0].cpu().detach().numpy()])


# In[146]:


test_data = cardio('', transform=None, mood='test')
test_dl = DataLoader(test_data, batch_size= 8)


# In[ ]:


# def evaluate(model, loader, device, criterion, mode='validation'):
mdl.eval()
# total_correct = 0
# total_loss = 0
# total = 0
# acc_list = list()
for i, images in enumerate(test_dl):
  images = images.to(device).float()
  # labels = labels.to(device).float()
  with torch.no_grad():
    outputs = mdl(images)
    # labels = labels.reshape(-1,1,320,320)
    # loss = criterion(outputs, labels)
    # total_loss += loss.item() * images.size(0)
    # total += images.size(0)
    # _, predictions = outputs.max(1)
    # total_correct += (labels == predictions).sum()
    outputs = outputs > 0.5
#     labels = labels.type(torch.BoolTensor)
#     acc_list += compute_accuracy(labels.to(device), outputs.to(device))
# accuracy = sum(acc_list) / len(acc_list)

# losss = total_loss / total
# # accuracy = total_correct / total
# print(f'Validation Loss({losss:6.4f}) Accuracy ({accuracy:6.4f})')


# In[159]:


mdl.eval()
for i, images in enumerate(test_dl):
  images = images.to(device).float()
  with torch.no_grad():
    outputs = mdl(images)
    outputs = outputs > 0.5
    if(i>110):
      break


# In[155]:


len(test_dl)


# In[ ]:


mdl.eval()
images = torch.tensor(data_test[500]).to(device).float().unsqueeze(0)
with torch.no_grad():
  output = mdl(images)
  output = outputs > 0.5


# In[152]:


outputs.shape


# In[160]:


vis([images[7][0].cpu().detach().numpy(),outputs[7][0].cpu().detach().numpy()])


# In[ ]:


(outputs > 0.5).cpu().detach().numpy().any()


# In[ ]:




