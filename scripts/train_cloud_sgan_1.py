import numpy as np
import torch
from torch import nn
from mapsgan import SGANSolver, data_loader
from sgan import TrajectoryGenerator, TrajectoryDiscriminator
import mapsgan.experiments as experiments
import time


mode = 'sgan'
fileprefix = '/cloud/sgan_1'
lr_gen = 1e-3
lr_dis = 1e-3

if torch.cuda.is_available():
    print('Cuda is available')
else:
    print('Cuda is NOT available')

print('Loading dataset...')
experiment = experiments.ETH() # we store filepaths and arguments in here
experiment.init_default_args() # those are some default SGAN parameters used in SGANSolver
print(experiment.best_k)
#experiment.k_best =
dataset, trainloader = data_loader(in_len=8, out_len=12, batch_size=4, num_workers=1, path=experiment.train_dir)

print('Setting models...')
generator = TrajectoryGenerator(obs_len=8,
                                pred_len=12,
                                embedding_dim=16,
                                encoder_h_dim=32,
                                decoder_h_dim=32,
                                mlp_dim=64,
                                num_layers=1,
                                noise_dim=(8,),
                                noise_type='gaussian',
                                noise_mix_type='global',
                                pooling_type='pool_net',
                                pool_every_timestep=1,
                                dropout=0,
                                bottleneck_dim=32,
                                neighborhood_size=2,
                                grid_size=8,
                                batch_norm=0)

discriminator = TrajectoryDiscriminator(obs_len=8,
                                        pred_len=12,
                                        embedding_dim=16,
                                        h_dim=64,
                                        mlp_dim=64,
                                        num_layers=1,
                                        dropout=0,
                                        batch_norm=0,
                                        d_type='local')

solver = SGANSolver(generator, discriminator, experiment=experiment, # pls read the code and docstrings to get the idea
                loss_fns={'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss},
                optims_args={'generator': {'lr': lr_gen}, 'discriminator': {'lr': lr_dis}})

print('Starting training...')
time_start = time.time()
solver.train(trainloader, epochs = 10000, checkpoint_every=100, steps = {'generator': 1, 'discriminator': 1},
                save_model=True, model_name=fileprefix, save_every=500, restore_checkpoint_from=None)
time_elapsed = (time.time() - time_start)/60
print('End of training. Duration: ' + str(time_elapsed))