import torch
from dataloader import *
from models import Model, post_process_disparity
from loss_original import *
from transforms_original import *
from torchviz import make_dot
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.misc
import skimage

import pandas as pd
disps_pp = pd.read_csv('data/output/disparities_pp.npy', encoding='ISO-8859-1')
print(disps_pp.shape)

#
# disp_l = [d[:,0,:,:].unsqueeze(1) for d in disps]
# left = np.squeeze(np.transpose(disp_l[0][0,:,:,:].cpu().detach().numpy(), (1,2,0)))
# plt.imshow(left)
# plt.show()
#
#
# print(x)


with torch.no_grad():
    img_1 = Image.open('img_1.jpg')
    [orig_height, orig_width] = img_1.size
    img_1 = test_transform(img_1).to('cuda:1')
    print(img_1.shape)
    disp_1 = model_test.model(img_1)
    disp_1 = post_process_disparity(disp_1[0][:,0,:,:].cpu().numpy())
    disp_to_img = skimage.transform.resize(disp_1.squeeze(), [orig_width, orig_height], mode='constant')
    #disp_to_img = np.array(Image.fromarray(disp_1.squeeze()).resize([orig_height, orig_width]))
    print(disp_to_img.shape)
    plt.imshow(disp_to_img, cmap='plasma')
    plt.show()


    img_2 = Image.open('img_2.jpg')
    [orig_height, orig_width] = img_2.size
    img_2 = test_transform(img_2).to('cuda:1')
    print(img_2.shape)
    disp_2 = model(img_2)
    disp_2 = post_process_disparity(disp_2[0][:,0,:,:].cpu().numpy())
    disp_to_img = skimage.transform.resize(disp_2.squeeze(), [orig_width, orig_height], mode='constant')

    #disp_to_img = np.array(Image.fromarray(disp_2.squeeze()).resize([orig_height, orig_width]))
    print(disp_to_img.shape)
    plt.imshow(disp_to_img, cmap='plasma')
    plt.show()