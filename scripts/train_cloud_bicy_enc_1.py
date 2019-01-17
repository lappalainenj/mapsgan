import torch
from torch import nn
from mapsgan import cVAESolver, cLRSolver, BicycleSolver, BicycleGenerator, ToyGenerator, ToyDiscriminator, data_loader
import mapsgan.experiments as experiments
import time


mode = 'cvae'
fileprefix = '/cloud/bicy_enc_1'
lr_gen = 1e-3
lr_dis = 1e-3

if torch.cuda.is_available():
    print('Cuda is available')
else:
    print('Cuda is NOT available')

print('Loading dataset...')
experiment = experiments.ETH() # we store filepaths and arguments in here
dataset, trainloader = data_loader(in_len=8, out_len=12, batch_size=64, num_workers=1, path=experiment.train_dir,
                                  shuffle=True)

print('Setting models...')
generator = BicycleGenerator(generator=ToyGenerator, start_mode=mode)
discriminator = ToyDiscriminator()
print('Start Mode: ' + generator.mode)

cvaesolver = cVAESolver(generator, discriminator,
                loss_fns={'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss},
                optims_args={'generator': {'lr': lr_gen}, 'discriminator': {'lr': lr_dis}})
clrsolver = cLRSolver(generator, discriminator,
                loss_fns={'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss},
                optims_args={'generator': {'lr': lr_gen}, 'discriminator': {'lr': lr_dis}})

solver = BicycleSolver(generator, discriminator, cvaesolver, clrsolver,
                loss_fns={'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss},
                optims_args={'generator': {'lr': lr_gen}, 'discriminator': {'lr': lr_dis}})

print('Starting training...')
time_start = time.time()
solver.train(trainloader, epochs = 10000, checkpoint_every=100, steps={'generator': 1, 'discriminator': 1},
              save_model=True, model_name=fileprefix, save_every=500, restore_checkpoint_from=None)
time_elapsed = (time.time() - time_start)/60
print('End of training. Duration: ' + str(time_elapsed) + 'mins')