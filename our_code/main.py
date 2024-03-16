import torch
from dataloader_rewritten import *
from models import Model
from loss_rewritten import *
from transforms_rewritten import *
from torchviz import make_dot

params = dict({'encoder': 'resnet',
               'dataset': 'kitti',
               'data_dir': 'data/kitti/',
               'train_filenames': 'data/filenames/kitti_train_files.txt',
               'val_filenames': 'data/filenames/kitti_val_files.txt',
               'test_filenames': 'data/filenames/kitti_test_files_simple.txt',
               'pretrained_dir': None,
               'input_height': 256,
               'input_width': 512,
               'batch_size': 8,
               'do_augmentation': True,
               'augment_parameters': [0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
               'n_epochs': 50,
               'lr': 1e-4,
               'lr_loss_weight': 1.0,
               'alpha_image_loss': 0.85,
               'disp_gradient_loss_weight': 0.1,
               'num_threads': 8,
               'model_path': 'models/',
               'output_dir': 'outputs/',
               'device': 'cuda:0',
               'num_workers': 8,
               'mode': 'train'
})
# other params: lr_loss_weight, alpha_image_loss, disp_grad_loss_weight,
# do_stereo, wrap_mode, use_deconv, num_gpus, num_threads

# dataset = MonoDataset('image_02/kitti/', 'image_02/filenames/kitti_train_files.txt', 'kitti', 'train')
# sample = dataset.__getitem__(1)
# print(len(dataset))
# sample['left_img'].show()
# sample['right_img'].show()
# disp = np.load('../MonoDepth-PyTorch-master/disparities.npy')
# print(disp.shape)
# disp_pp = np.load('../MonoDepth-PyTorch-master/disparities_pp.npy')
# print(disp_pp.shape)

# model = Model(params)
# for image_02 in model.train_loader:
#     left_imgs = image_02['left_img'].to(model.device)
#     break
# disps = model.model(left_imgs)
# dot = make_dot(disps, params = dict(model.model.named_parameters()))
# dot.format = 'png'
# dot.render('outputs/'+params['encoder'])

model = Model(params)
#model.train()
model.test()
#
# disp = np.load('outputs/kitti_resnet_disparities_old.npy')
# print(disp.shape)
# disp_pp = np.load('outputs/kitti_resnet_disparities_pp_old.npy')
# print(disp_pp.shape)