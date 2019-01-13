import torch
from torch import nn
from mapsgan import cVAESolver, BicycleGenerator, ToyGenerator, ToyDiscriminator, data_loader
import mapsgan.experiments as experiments
import time


if torch.cuda.is_available():
    print('Cuda is available')
else:
    print('Cuda is NOT available')

print('Loading dataset...')
experiment = experiments.ETH() # we store filepaths and arguments in here
dataset, trainloader = data_loader(in_len=8, out_len=12, batch_size=64, num_workers=1, path=experiment.train_dir,
                                  shuffle=True)

print('Setting models...')
generator = BicycleGenerator(generator=ToyGenerator, start_mode='cvae')
discriminator = ToyDiscriminator()
print('Mode: ' + generator.mode)

solver = cVAESolver(generator, discriminator,
                loss_fns={'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss},
                optims_args={'generator': {'lr': 1e-3}, 'discriminator': {'lr': 1e-3}})

print('Starting training...')
time_start = time.time()
solver.train(trainloader, epochs = 1, checkpoint_every=200, steps={'generator': 1, 'discriminator': 1},
              save_model=True, model_name='/cloud/cvae_1', save_every=1000, restore_checkpoint_from=None)
time_elapsed = (time.time() - time_start)/60
print('End of training. Duration: ' + str(time_elapsed))