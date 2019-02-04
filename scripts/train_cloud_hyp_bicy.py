import torch
from mapsgan import BicycleSolver, BicycleGenerator, ToyGenerator, ToyDiscriminator, data_loader
import mapsgan.experiments as experiments
import time
import datetime

mode = 'cvae'
fileprefix_ = '/cloud/hyp_bicy'
lr_gen = 1e-4
lr_dis = 1e-4
lr_enc = 1e-4
list_weights = [ #{'disc': 5, 'traj': 10, 'kl': 0.01, 'z': 0.5},
                 #{'disc': 10, 'traj': 10, 'kl': 10, 'z': 10},
                 {'disc': 1, 'traj': 1, 'kl': 1, 'z': 1}]


#loss_weights={'disc': 1, 'traj': 2, 'kl': 0.1, 'z': 0.5}

if torch.cuda.is_available():
    print('Cuda is available')
else:
    print('Cuda is NOT available')

print('Loading dataset...')
experiment = experiments.ETH() # we store filepaths and arguments in here
dataset, trainloader = data_loader(in_len=8, out_len=12, batch_size=64, num_workers=1,
                                   path=experiment.test_dir, shuffle=True)
_, testloader = data_loader(in_len=8, out_len=12, batch_size=1, num_workers=1, path=experiment.test_dir,
                                  shuffle=False)

for it, loss_weights in enumerate(list_weights):
    fileprefix = fileprefix_ + '_' + str(it)

    generator = BicycleGenerator(generator=ToyGenerator, start_mode=mode)
    discriminator = ToyDiscriminator()

    solver = BicycleSolver(generator, discriminator,
                    loss_weights=loss_weights,
                    optims_args={'generator': {'lr': lr_gen}, 'discriminator': {'lr': lr_dis}, 'encoder': {'lr': lr_enc}})

    print('\nIteration ', it + 1, '/', len(list_weights))
    print(datetime.datetime.now())
    time_start = time.time()
    solver.train(trainloader, epochs = 5000, checkpoint_every=9, print_every=9, val_every=False, testloader=testloader,
                 steps={'generator': 1, 'discriminator': 1},
                 save_model=True, model_name=fileprefix, save_every=10000, restore_checkpoint_from=None)
    time_elapsed = (time.time() - time_start)/60
    print('End of training. Duration: ' + str(time_elapsed) + 'mins\n-------------------------------------------------')
print('End of loop')