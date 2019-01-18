import random
import torch
from torch import nn
import numpy as np
import time
import os
from pathlib import Path
from mapsgan.utils import get_dtypes, relative_to_abs, init_weights, get_z_random
from mapsgan.losses import l2_loss as loss_fn_l2
from mapsgan.losses import kl_loss as loss_fn_kl

long_dtype, dtype = get_dtypes()  # dtype is either torch.FloatTensor or torch.cuda.FloatTensor
cuda = torch.cuda.is_available()
root_path = Path(os.path.realpath(__file__)).parent.parent  # basefolder of mapsgan git


class BaseSolver:
    """Abstract solver class.

    Args:
        models (dict): Dict storing 'generator' and 'discriminator' modules.
        optim (torch.optim.Optimizer object): Optimizer. Default is Adam.
        optims_args (dict): Dict storing 'generator' and 'discriminator' optim args.
            Default is {'lr': 1e-3} for both.
        loss_fns (dict): Dict storing 'norm' and 'gan' loss functions.
            Default is nn.L1Lossa and nn.BCEWithLogitsLoss.

    Attributes:
        All args.
        train_loss_history (dict): Storing all types of losses for 'generator' and 'discriminator'.
        test_loss_history (dict): Same as train_loss_history for test data.

    Important: All method here should be generic enough to serve all solvers.
    """

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None, loss_fns=None,
                 loss_weights=None, init_params=False):
        self.generator = generator
        self.discriminator = discriminator
        self.models = [self.generator, self.discriminator]
        if init_params:
            [model.apply(init_weights) for model in self.models]
        if cuda:
            [model.cuda() for model in self.models]
        self.optim = optim
        self.optims_args = optims_args if optims_args else {'generator': {'lr': 1e-3}, 'discriminator': {'lr': 1e-3}}  # default
        self.optimizer_g = None
        self.optimizer_d = None
        self.loss_fns = loss_fns if loss_fns else {'traj': nn.L1Loss, 'disc': nn.BCEWithLogitsLoss, 'z': nn.L1Loss,
                                                   'kl': loss_fn_kl}
        weights = {key:1. for key in self.loss_fns}
        self.loss_weights = loss_weights if loss_weights else weights
        self.train_loss_history = {'generator': {'G_BCE': [], 'G_L1': []},
                                   'discriminator': {'D_Real': [], 'D_Fake': []}}

    def save_checkpoint(self, trained_epochs, model_name):
        checkpoint = {'epochs': trained_epochs,
                      'g_state': self.generator.state_dict(),
                      'd_state': self.discriminator.state_dict(),
                      'g_optim_state': self.optimizer_g.state_dict(),
                      'd_optim_state': self.optimizer_d.state_dict(),
                      'train_loss_history': self.train_loss_history}
        self.model_str = 'models/' + model_name + '_' + time.strftime("%Y%m%d-%H%M%S") + '_epoch_' + str(trained_epochs)
        self.model_path = root_path / self.model_str
        torch.save(checkpoint, self.model_path)
        print('Training state saved to:\n' + str(self.model_path))

    def load_generator(self, model_path):
        if not cuda:
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path)
        self.generator.load_state_dict(checkpoint['g_state'])

    def load_checkpoint(self, model_path, init_optim=False):
        #print('Restoring from checkpoint')
        if not cuda:
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path)
        self.generator.load_state_dict(checkpoint['g_state'])
        self.discriminator.load_state_dict(checkpoint['d_state'])
        if init_optim:
            self.init_optimizers()
            self.optimizer_g.load_state_dict(checkpoint['g_optim_state'])
            self.optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        self.train_loss_history = checkpoint['train_loss_history']
        total_epochs = checkpoint['epochs']
        return total_epochs

    def train(self, loader, epochs, checkpoint_every=1, print_every=False, steps={'generator': 1, 'discriminator': 1},
              save_model=False, model_name='', save_every=1000, restore_checkpoint_from=None):
        """Trains the GAN.

        Args:
            loader: Dataloader.
            epochs (int): Number of epochs.
            checkpoint_every (int): Determines for which epochs a checkpoint is created.
            steps (dict): Steps to take on the same batch for generator and discriminator.
            save_model (bool): whether to save the model at the end of training
            restore_checkpoint_from (str): the path to load the model and optimizer states from to continue training

        Important: Keep generic to suit all Solvers.
        """
        self.init=True
        if restore_checkpoint_from is not None and os.path.isfile(restore_checkpoint_from):
            [prev_epochs] = \
                self.load_checkpoint(restore_checkpoint_from, init_optim=True)
            trained_epochs = prev_epochs
            print('Checkpoint restored')
        else:
            trained_epochs = 0
            self.init_optimizers()
            print('Training new model')


        self.generator.train()
        self.discriminator.train()
        if print_every:
            self._pprint(epochs, init=self.init)
        while epochs:
            gsteps = steps['generator']
            dsteps = steps['discriminator']
            for batch in loader:

                while dsteps:
                    losses_d = self.discriminator_step(batch, self.generator, self.discriminator, self.optimizer_d)
                    dsteps -= 1
                while gsteps:
                    losses_g = self.generator_step(batch, self.generator, self.discriminator, self.optimizer_g)
                    gsteps -= 1

            if epochs % checkpoint_every == 0:
                self._checkpoint(losses_g, losses_d)

            if print_every and (epochs % print_every == 0):
                self._pprint(epochs)

            trained_epochs += 1
            if (epochs % save_every == 0) & save_model:
                self.save_checkpoint(trained_epochs, model_name)

            epochs -= 1

        # end of training operations
        if save_model:
            self.save_checkpoint(trained_epochs, model_name)

    def test(self, loader, load_checkpoint_from=None, seed=17, z_dim=8, z_interpolation=None):
        """Tests the generator on unseen data.

        Args:
            loader: Dataloader.
            load_checkpoint_from (str): path to saved model
        """
        torch.manual_seed(seed)
        if load_checkpoint_from is not None and os.path.isfile(load_checkpoint_from):
            #print('Loading from checkpoint')
            if not cuda:
                checkpoint = torch.load(load_checkpoint_from, map_location='cpu')
            else:
                checkpoint = torch.load(load_checkpoint_from)
            self.generator.load_state_dict(checkpoint['g_state'])

        if cuda:
            self.generator.cuda()

        self.generator.eval()
        out = {'xy_in': [], 'xy_out': [], 'xy_pred': []}
        for batch in loader:
            if cuda:
                batch = {key: tensor.cuda() for key, tensor in batch.items()}
            xy_in = batch['xy_in']
            xy_out = batch['xy_out']
            dxdy_in = batch['dxdy_in']
            seq_start_end = batch['seq_start_end']
            if z_interpolation is None:
                z = get_z_random(xy_in.size(1), z_dim)
            else:
                z = z_interpolation
            dxdy_pred = self.generator(xy_in, dxdy_in, seq_start_end, user_noise=z)
            xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
            for seq in seq_start_end:
                start, end = seq
                out['xy_in'].append(xy_in[:, start:end].cpu().numpy())
                out['xy_out'].append(xy_out[:, start:end].cpu().numpy())
                out['xy_pred'].append(xy_pred[:, start:end].cpu().detach().numpy())
        return out

    def interpolate(self, loader, scene=25, stepsize=0.2, seed=20, z_dim=8, load_checkpoint_from=None):
        torch.manual_seed(seed)
        if load_checkpoint_from is not None and os.path.isfile(load_checkpoint_from):
            print('Loading from checkpoint')
            if not cuda:
                checkpoint = torch.load(load_checkpoint_from, map_location='cpu')
            else:
                checkpoint = torch.load(load_checkpoint_from)
            self.generator.load_state_dict(checkpoint['g_state'])

        if cuda:
            self.generator.cuda()

        self.generator.eval()
        out = {'xy_in': [], 'xy_out': [], 'xy_pred': []}
        batch = list(iter(loader))[scene]
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        xy_out = batch['xy_out']
        dxdy_in = batch['dxdy_in']
        seq_start_end = batch['seq_start_end']
        # Interpolation
        t=np.arange(0, 1.+stepsize, stepsize)
        z0 = get_z_random(xy_in.size(1), z_dim)
        z1 = get_z_random(xy_in.size(1), z_dim)
        for ti in t:
            z = z0 + ti*(z1-z0)
            dxdy_pred = self.generator(xy_in, dxdy_in, seq_start_end, user_noise=z)
            xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
            for seq in seq_start_end:
                start, end = seq
                out['xy_in'].append(xy_in[:, start:end].cpu().numpy())
                out['xy_out'].append(xy_out[:, start:end].cpu().numpy())
                out['xy_pred'].append(xy_pred[:, start:end].cpu().detach().numpy())
        return out



    def _checkpoint(self, losses_g, losses_d):
        """Checkpoint during training.

        Args:
            losses_g (tuple): Return of function generator_step.
            losses_d (tuple): Return of function discriminator_step.

        Note: Here we can store everything that we need for evaluation and
            save states and models to the hard drive.
        TODO: Implement a mechanism that saves the models whenever accuracy increased.
        """
        if hasattr(self.generator, 'mode'):
            if self.generator.mode == 'clr':
                self.train_loss_history['clr']['generator']['G_BCE'].append(losses_g[0])
                self.train_loss_history['clr']['generator']['G_L1'].append(losses_g[1])
                self.train_loss_history['clr']['discriminator']['D_Fake'].append(losses_d[0])
                self.train_loss_history['clr']['discriminator']['D_Real'].append(losses_d[1])
            elif self.generator.mode == 'cvae':
                self.train_loss_history['cvae']['generator']['G_BCE'].append(losses_g[0])
                self.train_loss_history['cvae']['generator']['G_L1'].append(losses_g[1])
                self.train_loss_history['cvae']['generator']['G_KL'].append(losses_g[2])
                self.train_loss_history['cvae']['discriminator']['D_Fake'].append(losses_d[0])
                self.train_loss_history['cvae']['discriminator']['D_Real'].append(losses_d[1])
        else:
            self.train_loss_history['generator']['G_BCE'].append(losses_g[0])
            self.train_loss_history['generator']['G_L1'].append(losses_g[1])
            self.train_loss_history['discriminator']['D_Fake'].append(losses_d[0])
            self.train_loss_history['discriminator']['D_Real'].append(losses_d[1])

    def _pprint(self, epochs, init=False):
        """Pretty prints the losses."""
        if hasattr(self.generator, 'mode'):
            if self.generator.mode == 'clr':
                loss_history = self.train_loss_history['clr']
            if self.generator.mode == 'cvae':
                loss_history = self.train_loss_history['cvae']
        else:
            loss_history = self.train_loss_history
        if init:
            msg = f"\n{'Generator Losses':>23}"
            msg += 'Discriminator Losses'.rjust(len(loss_history['generator']) * 10 + 4)
            msg += '\nEpochs '
            for type in loss_history['generator']:
                msg += f'{type:<10}'
            for type in loss_history['discriminator']:
                msg += f'{type:<10}'
        else:
            msg = f'{epochs:<7.0f}'
            for type, loss in loss_history['generator'].items():
                msg += f'{loss[-1]:<10.3f}'
            msg += ''
            for type, loss in loss_history['discriminator'].items():
                msg += f'{loss[-1]:<10.3f}'
        print(msg)

    def init_optimizers(self):
        self.optimizer_g = self.optim(self.generator.parameters(), **self.optims_args['generator'])
        self.optimizer_d = self.optim(self.discriminator.parameters(), **self.optims_args['discriminator'])


class Solver(BaseSolver):
    """Implements a generator and a discriminator step.

    See BaseSolver for detailed docstring.
    """

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None, loss_fns=None,
                 loss_weights=None, init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, loss_weights, init_params)

    def generator_step(self, batch, generator, discriminator, optimizer_g):
        """Generator optimization step.

        Args:
            batch: Batch from the data loader.
            generator: Generator module.
            discriminator: Discriminator module.
            optimizer_g: Generator optimizer.

        Returns:
            discriminator loss on fake
            norm loss on trajectory
            total generator loss
        """
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        dxdy_out = batch['dxdy_out']
        dxdy_in = batch['dxdy_in']
        seq_start_end = batch['seq_start_end']

        loss_fn_disc = self.loss_fns['disc']()  # discriminator loss
        loss_fn_traj = self.loss_fns['traj']()  # comparing trajectories
        w_disc = self.loss_weights['disc']
        w_traj = self.loss_weights['traj']

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        scores_fake = discriminator(xy_fake, seq_start_end)
        target_fake = torch.ones_like(scores_fake).type(dtype) * random.uniform(0.7, 1.2)

        traj_loss = loss_fn_traj(dxdy_pred, dxdy_out)  # IMPORTANT: Indeed SGAN compares displacements!
        disc_loss = loss_fn_disc(scores_fake, target_fake)

        loss = w_disc * disc_loss + w_traj * traj_loss
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        return disc_loss.item(), traj_loss.item()

    def discriminator_step(self, batch, generator, discriminator, optimizer_d):
        """Discriminator optimization step.

        Args:
            batch: Batch from the data loader.
            generator: Generator module.
            discriminator: Discriminator module.
            optimizer_d: Discriminator optimizer.

        Returns:
            discriminator loss on fake
            discriminator loss on real
            total discriminator loss
        """
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        xy_out = batch['xy_out']
        dxdy_in = batch['dxdy_in']
        seq_start_end = batch['seq_start_end']
        loss_fn_disc = self.loss_fns['disc']()

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])

        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        xy_real = torch.cat([xy_in, xy_out], dim=0)

        scores_fake = discriminator(xy_fake, seq_start_end)
        scores_real = discriminator(xy_real, seq_start_end)

        target_real = torch.ones_like(scores_real).type(dtype) * random.uniform(0.7, 1.2)
        target_fake = torch.zeros_like(scores_fake).type(dtype) * random.uniform(0., 0.3)

        disc_loss_real = loss_fn_disc(scores_real, target_real)
        disc_loss_fake = loss_fn_disc(scores_fake, target_fake)

        loss = disc_loss_fake + disc_loss_real
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        return disc_loss_fake.item(), disc_loss_real.item()


class SGANSolver(BaseSolver):
    """Implements a generator and a discriminator step for SGAN.

    See BaseSolver for detailed docstring.

    Important note: For now we store the many arguments used in SGAN in the experiment object.
    """

    def __init__(self, generator, discriminator, experiment, optim=torch.optim.Adam, optims_args=None, loss_fns=None,
                 loss_weights=None, init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, loss_weights, init_params)
        self.args = experiment

    def generator_step(self, batch, generator, discriminator, optimizer_g):
        """Generator optimization step.

        Args:
            batch: Batch from the data loader.
            generator: Generator module.
            discriminator: Discriminator module.
            optimizer_g: Generator optimizer.

        Returns:
            discriminator loss on fake
            norm loss on trajectory
            total generator loss
        """
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        dxdy_out = batch['dxdy_out']
        dxdy_in = batch['dxdy_in']
        loss_mask = batch['loss_mask'][:, self.args.in_len:]
        seq_start_end = batch['seq_start_end']

        loss_fn_disc = self.loss_fns['disc']()
        w_disc = self.loss_weights['disc']

        # Important: This is for their diversity loss, cp. paragraph 3.5.
        l2_losses = []

        for _ in range(self.args.best_k):
            dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
            xy_pred = relative_to_abs(dxdy_in, xy_in[-1])

            if self.args.weight_l2_loss:  # Important: Computes loss on displacements.
                l2_losses.append(self.args.weight_l2_loss * loss_fn_l2(dxdy_pred, dxdy_out, loss_mask, mode='raw'))

        l2_loss = 0
        if self.args.weight_l2_loss:
            l2_losses = torch.stack(l2_losses, dim=1)
            l2_losses_sum = 0
            for start, end in seq_start_end.data:
                _l2_losses = l2_losses[start:end]
                _l2_losses = torch.sum(_l2_losses, dim=0)
                _l2_losses = torch.min(_l2_losses) / torch.sum(loss_mask[start:end])
                l2_losses_sum += _l2_losses
            l2_loss = l2_losses_sum

        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        dxdy_fake = torch.cat([dxdy_in, dxdy_pred], dim=0)
        scores_fake = discriminator(xy_fake, dxdy_fake, seq_start_end)
        target_fake = torch.ones_like(scores_fake).type(dtype) * random.uniform(0.7, 1.2)

        disc_loss = loss_fn_disc(scores_fake, target_fake)

        loss = w_disc * disc_loss + l2_loss
        optimizer_g.zero_grad()
        loss.backward()
        if self.args.clip:
            # QUESTION: Is this a Lipschitz constraining method as in W-GAN?
            nn.utils.clip_grad_norm_(generator.parameters(), self.args.clip)
        optimizer_g.step()

        return disc_loss.item(), l2_loss.item()

    def discriminator_step(self, batch, generator, discriminator, optimizer_d):
        """Discriminator optimization step.

        Args:
            batch: Batch from the data loader.
            generator: Generator module.
            discriminator: Discriminator module.
            optimizer_d: Discriminator optimizer.

        Returns:
            discriminator loss on fake
            discriminator loss on real
            total discriminator loss
        """
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        xy_out = batch['xy_out']
        dxdy_in = batch['dxdy_in']
        dxdy_out = batch['dxdy_out']
        seq_start_end = batch['seq_start_end']
        loss_fn_disc = self.loss_fns['disc']()

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])

        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        xy_real = torch.cat([xy_in, xy_out], dim=0)
        dxdy_real = torch.cat([dxdy_in, dxdy_out], dim=0)
        dxdy_fake = torch.cat([dxdy_in, dxdy_pred], dim=0)

        # IMPORTANT: In SGAN the discriminator gets the displacement, too.
        scores_fake = discriminator(xy_fake, dxdy_fake, seq_start_end)
        scores_real = discriminator(xy_real, dxdy_real, seq_start_end)

        target_real = torch.ones_like(scores_real).type(dtype) * random.uniform(0.7, 1.2)
        target_fake = torch.zeros_like(scores_fake).type(dtype) * random.uniform(0., 0.3)

        disc_loss_real = loss_fn_disc(scores_real, target_real)
        disc_loss_fake = loss_fn_disc(scores_fake, target_fake)

        loss = disc_loss_fake + disc_loss_real

        optimizer_d.zero_grad()
        loss.backward()
        if self.args.clip:
            # QUESTION: Is this a Lipschitz constraining method as in W-GAN?
            nn.utils.clip_grad_norm_(discriminator.parameters(), self.args.clip)
        optimizer_d.step()

        return disc_loss_fake.item(), disc_loss_real.item()


class cLRSolver(Solver):
    """Implements a generator and a discriminator step for the conditional latent regressor model.

    See BaseSolver for detailed docstring.
    # TODO: Either make generic somehow to work with sgan too or implement an sgan version.
    """

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None,
                 loss_fns=None, loss_weights=None, init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, loss_weights, init_params)
        generator.clr()
        self.train_loss_history = {'clr': {'generator': {'G_BCE': [], 'G_L1': []},
                                           'discriminator': {'D_Real': [], 'D_Fake': []}}}

    def generator_step(self, batch, generator, discriminator, optimizer_g):
        """Generator optimization step.

        Args:
            batch: Batch from the data loader.
            generator: Generator module.
            discriminator: Discriminator module.
            optimizer_g: Generator optimizer.

        Returns:
            discriminator loss on fake
            norm loss on trajectory
            total generator loss
        """
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        # dxdy_out = batch['dxdy_out']
        dxdy_in = batch['dxdy_in']
        seq_start_end = batch['seq_start_end']

        loss_fn_disc = self.loss_fns['disc']()
        loss_fn_z = self.loss_fns['z']()
        w_disc = self.loss_weights['disc']
        #w_traj = self.loss_weights['traj']
        w_z = self.loss_weights['z']

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        scores_fake = discriminator(xy_fake, seq_start_end)
        target_fake = torch.ones_like(scores_fake).type(dtype) * random.uniform(0.7, 1.2)

        z_loss = loss_fn_z(generator.mu, generator.z_random)  # Important: Here, we compare the latent vectors.
        disc_loss = loss_fn_disc(scores_fake, target_fake)

        loss = w_disc * disc_loss + w_z * z_loss
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        return disc_loss.item(), z_loss.item()


class cVAESolver(Solver):
    """Implements a generator and a discriminator step for the conditional latent regressor model.

    See BaseSolver for detailed docstring.
    # TODO: Either make generic somehow to work with sgan too or implement an sgan version.
    """

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None,
                 loss_fns=None, loss_weights=None, init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, loss_weights, init_params)
        generator.cvae()
        self.train_loss_history = {'cvae': {'generator': {'G_BCE': [], 'G_L1': [], 'G_KL': []},
                                            'discriminator': {'D_Real': [], 'D_Fake': []}}}

    def generator_step(self, batch, generator, discriminator, optimizer_g):
        """Generator optimization step.

        Args:
            batch: Batch from the data loader.
            generator: Generator module.
            discriminator: Discriminator module.
            optimizer_g: Generator optimizer.

        Returns:
            discriminator loss on fake
            norm loss on trajectory
            total generator loss
        """
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        xy_out = batch['xy_out']
        dxdy_out = batch['dxdy_out']
        dxdy_in = batch['dxdy_in']
        seq_start_end = batch['seq_start_end']

        loss_fn_disc = self.loss_fns['disc']()
        loss_fn_traj = self.loss_fns['traj']()
        loss_fn_kl = self.loss_fns['kl']
        w_disc = self.loss_weights['disc']
        w_traj = self.loss_weights['traj']
        w_kl = self.loss_weights['kl']

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end, xy_out)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        scores_fake = discriminator(xy_fake, seq_start_end)
        target_fake = torch.ones_like(scores_fake).type(dtype) * random.uniform(0.7, 1.2)

        traj_loss = loss_fn_traj(dxdy_pred, dxdy_out)
        disc_loss = loss_fn_disc(scores_fake, target_fake)
        kl_loss = loss_fn_kl(generator.mu, generator.logvar)

        loss = w_disc * disc_loss + w_traj * traj_loss + w_kl * kl_loss
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        return disc_loss.item(), traj_loss.item(), kl_loss.item()

    def discriminator_step(self, batch, generator, discriminator, optimizer_d):
        """Discriminator optimization step.

        Args:
            batch: Batch from the data loader.
            generator: Generator module.
            discriminator: Discriminator module.
            optimizer_d: Discriminator optimizer.

        Returns:
            discriminator loss on fake
            discriminator loss on real
            total discriminator loss
        """
        if cuda:
            batch = {key: tensor.cuda() for key, tensor in batch.items()}
        xy_in = batch['xy_in']
        xy_out = batch['xy_out']
        dxdy_in = batch['dxdy_in']
        seq_start_end = batch['seq_start_end']
        loss_fn_disc = self.loss_fns['disc']()

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end, xy_out)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])

        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        xy_real = torch.cat([xy_in, xy_out], dim=0)

        scores_fake = discriminator(xy_fake, seq_start_end)
        scores_real = discriminator(xy_real, seq_start_end)

        target_real = torch.ones_like(scores_real).type(dtype) * random.uniform(0.7, 1.2)
        target_fake = torch.zeros_like(scores_fake).type(dtype) * random.uniform(0., 0.3)

        disc_loss_real = loss_fn_disc(scores_real, target_real)
        disc_loss_fake = loss_fn_disc(scores_fake, target_fake)

        loss = disc_loss_fake + disc_loss_real
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        return disc_loss_fake.item(), disc_loss_real.item()


class BicycleSolver(BaseSolver):

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None,
                 loss_fns=None, loss_weights=None, init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, loss_weights, init_params)

        self.cvaesolver = cVAESolver(generator, discriminator, optim, optims_args, loss_fns, loss_weights, init_params)
        self.clrsolver = cLRSolver(generator, discriminator, optim, optims_args, loss_fns, loss_weights, init_params)
        self.train_loss_history = {'clr': {'generator': {'G_BCE': [-1], 'G_L1': []},
                                           'discriminator': {'D_Real': [], 'D_Fake': []}},
                                   'cvae': {'generator': {'G_BCE': [], 'G_L1': [], 'G_KL': []},
                                            'discriminator': {'D_Real': [], 'D_Fake': []}}}

    def generator_step(self, batch, generator, discriminator, optimizer_g):
        if self.init:
            self.optimizer_g = self.optim([{'params': generator.generator.parameters()},
                                      {'params': generator.encoder.parameters()}],
                                     **self.optims_args['generator'])
            self.init=False
        if generator.mode == 'clr':
            self.optimizer_g.param_groups[1]['lr'] = 0.
            bce_loss, norm_loss = self.clrsolver.generator_step(batch, generator, discriminator, optimizer_g)
            losses = (bce_loss, norm_loss)
            generator.cvae()
        elif generator.mode == 'cvae':
            self.optimizer_g.param_groups[1]['lr'] = self.optimizer_g.defaults['lr']
            bce_loss, norm_loss, kl_loss = self.cvaesolver.generator_step(batch, generator, discriminator, optimizer_g)
            losses = (bce_loss, norm_loss, kl_loss)
            generator.clr()
        else:
            raise AssertionError('Mode must be either clr or cvae.')
        return losses

    def discriminator_step(self, batch, generator, discriminator, optimizer_d):
        if generator.mode == 'clr':
            loss_fake, loss_real = self.clrsolver.discriminator_step(batch, generator, discriminator, optimizer_d)
        elif generator.mode == 'cvae':
            loss_fake, loss_real = self.cvaesolver.discriminator_step(batch, generator, discriminator, optimizer_d)
        else:
            raise AssertionError('Mode must be either clr or cvae.')
        return loss_fake, loss_real

    def _checkpoint(self, losses_g, losses_d):
        """Checkpoint during training.

        Args:
            losses_g (tuple): Return of function generator_step.
            losses_d (tuple): Return of function discriminator_step.

        Note: Here we can store everything that we need for evaluation and
            save states and models to the hard drive.
        TODO: Implement a mechanism that saves the models whenever accuracy increased.
        """
        if self.generator.mode == 'cvae':
            self.train_loss_history['clr']['generator']['G_BCE'].append(losses_g[0])
            self.train_loss_history['clr']['generator']['G_L1'].append(losses_g[1])
            self.train_loss_history['clr']['discriminator']['D_Fake'].append(losses_d[0])
            self.train_loss_history['clr']['discriminator']['D_Real'].append(losses_d[1])
        elif self.generator.mode == 'clr':
            self.train_loss_history['cvae']['generator']['G_BCE'].append(losses_g[0])
            self.train_loss_history['cvae']['generator']['G_L1'].append(losses_g[1])
            self.train_loss_history['cvae']['generator']['G_KL'].append(losses_g[2])
            self.train_loss_history['cvae']['discriminator']['D_Fake'].append(losses_d[0])
            self.train_loss_history['cvae']['discriminator']['D_Real'].append(losses_d[1])

    def _pprint(self, epochs, init=False):
        """Pretty prints the losses."""
        if init:
            msg = f"\n{'Generator Losses':>23}"
            msg += 'Discriminator Losses'.rjust(len(self.train_loss_history['cvae']['generator']) * 10 + 4)
            msg += '\nEpochs '
            for type in self.train_loss_history['cvae']['generator']:
                msg += f'{type:<10}'
            for type in self.train_loss_history['cvae']['discriminator']:
                msg += f'{type:<10}'
        else:
            msg = f'{epochs:<7.0f}'
            for type, loss in self.train_loss_history['clr']['generator'].items():
                msg += f'{loss[-1]:<10.3f}' if loss else ''
            msg += ''.rjust(10)
            for type, loss in self.train_loss_history['clr']['discriminator'].items():
                msg += f'{loss[-1]:<10.3f}' if loss else ''
            msg += f'\t cLR\n'
            msg += f'{epochs:<7.0f}'
            for type, loss in self.train_loss_history['cvae']['generator'].items():
                msg += f'{loss[-1]:<10.3f}' if loss else ''
            msg += ''
            for type, loss in self.train_loss_history['cvae']['discriminator'].items():
                msg += f'{loss[-1]:<10.3f}' if loss else ''
            msg += f'\t cVAE'
        print(msg)

    def load_checkpoint(self, model_path, init_optim=False):
        #print('Restoring from checkpoint')
        if not cuda:
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path)
        self.generator.load_state_dict(checkpoint['g_state'])
        self.discriminator.load_state_dict(checkpoint['d_state'])
        self.optimizer_g = self.optim([{'params': generator.generator.parameters()},
                                       {'params': generator.encoder.parameters()}],
                                      **self.optims_args['generator'])
        if init_optim:
            self.optimizer_g = self.optim([{'params': generator.generator.parameters()},
                                           {'params': generator.encoder.parameters()}],
                                          **self.optims_args['generator'])
            self.optimizer_d = self.optim(self.discriminator.parameters(), **self.optims_args['discriminator'])

            self.optimizer_g.load_state_dict(checkpoint['g_optim_state'])
            self.optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        self.train_loss_history = checkpoint['train_loss_history']
        total_epochs = checkpoint['epochs']
        self.init = False
        return total_epochs
