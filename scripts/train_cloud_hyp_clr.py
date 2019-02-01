import torch
from torch import nn
from mapsgan import cLRSolver, BicycleGenerator, ToyGenerator, ToyDiscriminator, data_loader
import mapsgan.experiments as experiments
import time
import datetime

mode = 'clr'
fileprefix = '/cloud/hyp_clr'

list_lr = [1e-2, 1e-4, 1e-6, 1e-7]

if torch.cuda.is_available():
    print('Cuda is available')
else:
    print('Cuda is NOT available')

print('Loading dataset...')
experiment = experiments.ETH() # we store filepaths and arguments in here
dataset, trainloader = data_loader(in_len=8, out_len=12, batch_size=64, num_workers=1, path=experiment.train_dir,
                                  shuffle=True)
_, testloader = data_loader(in_len=8, out_len=12, batch_size=1, num_workers=1, path=experiment.test_dir,
                                  shuffle=False)

for it, lr in enumerate(list_lr):
    lr_gen = lr
    lr_dis = lr
    lr_enc = lr

    #print('Setting models...')
    generator = BicycleGenerator(generator=ToyGenerator, start_mode=mode)
    discriminator = ToyDiscriminator()

    solver = cLRSolver(generator, discriminator,
                    #loss_fns={'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss},
                    optims_args={'generator': {'lr': lr_gen}, 'discriminator': {'lr': lr_dis}, 'encoder': {'lr': lr_enc}})

    #print('Starting training...')
    print('\nIteration ', it + 1, '/', len(list_lr))
    print(datetime.datetime.now())
    time_start = time.time()
    solver.train(trainloader, epochs = 5000, checkpoint_every=10, print_every=100, val_every=10, testloader=testloader,
                  steps={'generator': 1, 'discriminator': 1},
                  save_model=True, model_name=fileprefix, save_every=10000, restore_checkpoint_from=None)
    time_elapsed = (time.time() - time_start)/60
    print('End of training. Duration: ' + str(time_elapsed) + 'mins\n-------------------------------------------------')
print('End of loop')