import numpy as np
import torch
from torch import nn
from mapsgan import Solver, ToyGenerator, ToyDiscriminator, data_loader
import mapsgan.experiments as experiments

mode = 'toymodel'
fileprefix = '/cloud/toymodel_1'
lr_gen = 1e-2
lr_dis = 1e-2

if torch.cuda.is_available():
    print('Cuda is available')
else:
    print('Cuda is NOT available')

print('Loading dataset...')
experiment = experiments.ETH() # we store filepaths and arguments in here
dataset, trainloader = data_loader(in_len=8, out_len=12, batch_size=64, num_workers=1, path=experiment.train_dir,)

print('Setting models...')
generator = ToyGenerator(in_len=8, out_len=12, noise_dim=(8,), decoder_h_dim=74)
discriminator = ToyDiscriminator()
print('Mode: ' + mode)

solver = Solver(generator, discriminator,
                loss_fns={'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss},
                optims_args={'generator': {'lr': lr_gen}, 'discriminator': {'lr': lr_dis}})

rint('Starting training...')
time_start = time.time()
solver.train(trainloader, epochs = 10000, checkpoint_every=50, save_model=True, model_name=fileprefix, save_every=500,
             steps = {'generator': 1, 'discriminator': 1})
time_elapsed = (time.time() - time_start)/60
print('End of training. Duration: ' + str(time_elapsed) + 'mins')