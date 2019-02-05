import torch
from mapsgan import BicycleSolver, BicycleGenerator, data_loader, get_sgan_generator
import mapsgan.experiments as experiments
import time
from mapsgan.sgan import TrajectoryGenerator, TrajectoryDiscriminator

cuda = torch.cuda.is_available()
ml = 'cpu' if not cuda else None

### Saved SGAN Models
model_path = '../../sgan/sgan/models/sgan-p-models/'
model = 'eth_12_model.pt'
###
checkpoint = torch.load(model_path + model, map_location=ml)

mode = 'cvae'
fileprefix = '/cloud/sgan_bicy_1'
lr_gen = 1e-3
lr_dis = 1e-3
lr_enc = 1e-3
loss_weights = {'disc': 1, 'traj': 2, 'kl': 0.1, 'z': 0.5}
obs_len = 8,
pred_len = 12
embedding_dim = 16
encoder_h_dim = 32
decoder_h_dim = 32
mlp_dim = 64
num_layers = 1
noise_dim = (8,)
noise_type = 'gaussian'
noise_mix_type = 'global'
pooling_type = 'pool_net'

if cuda:
    print('Cuda is available')
else:
    print('Cuda is NOT available')

print('Loading dataset...')
experiment = experiments.ETH()  # we store filepaths and arguments in here
dataset, trainloader = data_loader(in_len=8,
                                   out_len=12,
                                   batch_size=8,
                                   num_workers=1,
                                   path=experiment.train_dir,
                                   shuffle=True)

_, testloader = data_loader(in_len=8,
                            out_len=12,
                            batch_size=1,
                            num_workers=1,
                            path=experiment.test_dir,
                            shuffle=False)

print('Setting models...')
generator_sgan = get_sgan_generator(checkpoint)
generator = BicycleGenerator(generator=generator_sgan,
                             start_mode=mode,
                             embedding_dim=embedding_dim,
                             h_dim=decoder_h_dim - 8,
                             z_dim=8,
                             in_len=obs_len,
                             out_len=pred_len,
                             noise_type=noise_type,
                             noise_mix_type=noise_mix_type)

discriminator = TrajectoryDiscriminator(obs_len=8,
                                        pred_len=12,
                                        embedding_dim=16,
                                        h_dim=64,
                                        mlp_dim=64,
                                        num_layers=1,
                                        dropout=0,
                                        batch_norm=0,
                                        d_type='local')
print('Start Mode: ' + generator.mode)

solver = BicycleSolver(generator, discriminator,
                       loss_weights=loss_weights,
                       optims_args={'generator': {'lr': lr_gen},
                                    'discriminator': {'lr': lr_dis},
                                    'encoder': {'lr': lr_enc}})

print('Starting training...')
time_start = time.time()
solver.train(trainloader,
             epochs=10000,
             checkpoint_every=9,
             print_every=9,
             val_every=9,
             testloader=testloader,
             steps={'generator': 1, 'discriminator': 1},
             save_model=True,
             model_name=fileprefix,
             save_every=499,
             restore_checkpoint_from=None)
time_elapsed = (time.time() - time_start) / 60
print('End of training. Duration: ' + str(time_elapsed) + 'mins')
