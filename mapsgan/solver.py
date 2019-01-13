import random
import torch
from torch import nn
import numpy as np
import time
import os
from pathlib import Path
from mapsgan.utils import get_dtypes, relative_to_abs, init_weights
from mapsgan.losses import l2_loss as loss_fn_l2
from mapsgan.losses import kl_loss as loss_fn_kl

long_dtype, dtype = get_dtypes()  # dtype is either torch.FloatTensor or torch.cuda.FloatTensor
cuda = torch.cuda.is_available()
root_path = Path(os.path.realpath(__file__)).parent.parent # basefolder of mapsgan git


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
                 init_params=False):
        self.generator = generator
        self.discriminator = discriminator
        self.models = [self.generator, self.discriminator]
        if init_params:
            [model.apply(init_weights) for model in self.models]
        if cuda:
            [model.cuda() for model in self.models]
        self.optim = optim
        self.optims_args = optims_args
        if optims_args:
            self.optims_args = optims_args
        else:
            self.optims_args = {'generator': {'lr': 1e-3}, 'discriminator': {'lr': 1e-3}}  # default
        if loss_fns:
            self.loss_fns = loss_fns
        else:
            self.loss_fns = {'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss}  # default TODO: Add KL-DIV from utils
        self._reset_histories()

    def save_checkpoint(self, trained_epochs, optimizer_g, optimizer_d):
        checkpoint = { 'epochs':trained_epochs,
                       'g_state':self.generator.state_dict(),
                       'd_state':self.discriminator.state_dict(),
                       'g_optim_state':optimizer_g.state_dict(),
                       'd_optim_state':optimizer_d.state_dict()  }
        self.model_str = 'models/' + time.strftime("%Y%m%d-%H%M%S")  # save as time (dont overwrite others)
        self.model_path = root_path / self.model_str
        torch.save(checkpoint, self.model_path)
        print('Training state saved to:\n' + str(self.model_path))

    def load_checkpoint(self, model_path, optimizer_g, optimizer_d):
        print('Restoring from checkpoint')
        checkpoint = torch.load(model_path)
        self.generator.load_state_dict(checkpoint['g_state'])
        self.discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        total_epochs = checkpoint['epochs']
        return optimizer_g, optimizer_d, total_epochs

    def train(self, loader, epochs, checkpoint_every=1, steps={'generator': 1, 'discriminator': 1},
              save_model=False, restore_checkpoint_from=None):
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
        optimizer_g = self.optim(self.generator.parameters(), **self.optims_args['generator'])
        optimizer_d = self.optim(self.discriminator.parameters(), **self.optims_args['discriminator'])

        if restore_checkpoint_from is not None and os.path.isfile(restore_checkpoint_from):
            [optimizer_g, optimizer_d, prev_epochs] = \
                self.load_checkpoint(restore_checkpoint_from, optimizer_g, optimizer_d)
            trained_epochs = epochs + prev_epochs
            print('Checkpoint restored')
        else:
            trained_epochs = epochs
            print('Training new model')

        self.generator.train()
        self.discriminator.train()
        self._pprint(epochs, init=True)
        while epochs:
            gsteps = steps['generator']
            dsteps = steps['discriminator']
            for batch in loader:

                while dsteps:
                    losses_d = self.discriminator_step(batch, self.generator, self.discriminator, optimizer_d)
                    dsteps -= 1
                while gsteps:
                    losses_g = self.generator_step(batch, self.generator, self.discriminator, optimizer_g)
                    gsteps -= 1

            if epochs % checkpoint_every == 0:
                self._checkpoint(losses_g, losses_d)
                self._pprint(epochs)

            epochs -= 1

        # end of training operations
        if save_model:
            self.save_checkpoint(trained_epochs, optimizer_g, optimizer_d)


    def test(self, loader, load_checkpoint_from=None):
        """Tests the generator on unseen data.

        Args:
            loader: Dataloader.
            load_checkpoint_from (str): path to saved model
        """
        if load_checkpoint_from is not None and os.path.isfile(load_checkpoint_from):
            print('Loading from checkpoint')
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
            dxdy_pred = self.generator(xy_in, dxdy_in, seq_start_end)
            xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
            for seq in seq_start_end:
                start, end = seq
                out['xy_in'].append(xy_in[:, start:end].cpu().numpy())
                out['xy_out'].append(xy_out[:, start:end].cpu().numpy())
                out['xy_pred'].append(xy_pred[:, start:end].cpu().detach().numpy())
        return out

    def _reset_histories(self):
        self.train_loss_history = {'generator': {'G_gan': [], 'G_norm': [], 'G_total': []},
                                   'discriminator': {'D_real': [], 'D_fake': [], 'D_total': []}}
        self.train_acc_history = {}
        self.test_loss_history = {'generator': [], 'discriminator': []}
        self.test_acc_history = {}
        self.best_acc = 0

    def _checkpoint(self, losses_g, losses_d):
        """Checkpoint during training.

        Args:
            losses_g (tuple): Return of function generator_step.
            losses_d (tuple): Return of function discriminator_step.

        Note: Here we can store everything that we need for evaluation and
            save states and models to the hard drive.
        TODO: Implement a mechanism that saves the models whenever accuracy increased.
        """
        self.train_loss_history['generator']['G_gan'].append(losses_g[0])
        self.train_loss_history['generator']['G_norm'].append(losses_g[1])
        self.train_loss_history['generator']['G_total'].append(losses_g[2])
        self.train_loss_history['discriminator']['D_fake'].append(losses_d[0])
        self.train_loss_history['discriminator']['D_real'].append(losses_d[1])
        self.train_loss_history['discriminator']['D_total'].append(losses_d[2])

    def _pprint(self, epochs, init=False):
        """Pretty prints the losses."""
        if init:
            msg = f"\n{'Generator Losses':>23}"
            msg += 'Discriminator Losses'.rjust(len(self.train_loss_history['generator'])*10+4)
            msg += '\nEpochs '
            for type in self.train_loss_history['generator']:
                msg += f'{type:<10}'
            for type in self.train_loss_history['discriminator']:
                msg += f'{type:<10}'
        else:
            msg = f'{epochs:<7.0f}'
            for type, loss in self.train_loss_history['generator'].items():
                msg += f'{loss[-1]:<10.3f}'
            msg += ''
            for type, loss in self.train_loss_history['discriminator'].items():
                msg += f'{loss[-1]:<10.3f}'
        print(msg)


class Solver(BaseSolver):
    """Implements a generator and a discriminator step.

    See BaseSolver for detailed docstring.
    """

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None, loss_fns=None,
                 init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, init_params)

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

        loss_fn_gan = self.loss_fns['gan']()
        loss_fn_norm = self.loss_fns['norm']()

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        scores_fake = discriminator(xy_fake, seq_start_end)
        target_fake = torch.ones_like(scores_fake).type(dtype) * random.uniform(0.7, 1.2)

        norm_loss = loss_fn_norm(dxdy_pred, dxdy_out)  # IMPORTANT: Indeed SGAN compares displacements!
        gan_loss = loss_fn_gan(scores_fake, target_fake)

        loss = gan_loss + norm_loss
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        return gan_loss.item(), norm_loss.item(), loss.item()

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
        loss_fn_gan = self.loss_fns['gan']()

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])

        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        xy_real = torch.cat([xy_in, xy_out], dim=0)

        scores_fake = discriminator(xy_fake, seq_start_end)
        scores_real = discriminator(xy_real, seq_start_end)

        target_real = torch.ones_like(scores_real).type(dtype) * random.uniform(0.7, 1.2)
        target_fake = torch.zeros_like(scores_fake).type(dtype) * random.uniform(0., 0.3)

        gan_loss_real = loss_fn_gan(scores_real, target_real)
        gan_loss_fake = loss_fn_gan(scores_fake, target_fake)

        loss = gan_loss_fake + gan_loss_real
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        return gan_loss_fake.item(), gan_loss_real.item(), loss.item()


class SGANSolver(BaseSolver):
    """Implements a generator and a discriminator step for SGAN.

    See BaseSolver for detailed docstring.

    Important note: For now we store the many arguments used in SGAN in the experiment object.
    """

    def __init__(self, generator, discriminator, experiment, optim=torch.optim.Adam, optims_args=None, loss_fns=None,
                 init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, init_params)
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

        loss_fn_gan = self.loss_fns['gan']()

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

        gan_loss = loss_fn_gan(scores_fake, target_fake)

        loss = gan_loss + l2_loss
        optimizer_g.zero_grad()
        loss.backward()
        if self.args.clip:
            # QUESTION: Is this a Lipschitz constraining method as in W-GAN?
            nn.utils.clip_grad_norm_(generator.parameters(), self.args.clip)
        optimizer_g.step()

        return gan_loss.item(), l2_loss.item(), loss.item()

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
        loss_fn_gan = self.loss_fns['gan']()

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

        gan_loss_real = loss_fn_gan(scores_real, target_real)
        gan_loss_fake = loss_fn_gan(scores_fake, target_fake)

        loss = gan_loss_fake + gan_loss_real

        optimizer_d.zero_grad()
        loss.backward()
        if self.args.clip:
            # QUESTION: Is this a Lipschitz constraining method as in W-GAN?
            nn.utils.clip_grad_norm_(discriminator.parameters(), self.args.clip)
        optimizer_d.step()

        return gan_loss_fake.item(), gan_loss_real.item(), loss.item()


class cLRSolver(Solver):
    """Implements a generator and a discriminator step for the conditional latent regressor model.

    See BaseSolver for detailed docstring.
    # TODO: Either make generic somehow to work with sgan too or implement an sgan version.
    """

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None,
                 lambda_z = 1, loss_fns=None, init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, init_params)
        generator.clr()
        self.lambda_z = lambda_z

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

        loss_fn_gan = self.loss_fns['gan']()
        loss_fn_norm = self.loss_fns['norm']()
        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        scores_fake = discriminator(xy_fake, seq_start_end)
        target_fake = torch.ones_like(scores_fake).type(dtype) * random.uniform(0.7, 1.2)

        norm_loss = loss_fn_norm(generator.mu, generator.z_random)  # Important: Here, we compare the latent vectors.
        gan_loss = loss_fn_gan(scores_fake, target_fake)

        loss = gan_loss + norm_loss * self.lambda_z
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        return gan_loss.item(), norm_loss.item(), loss.item()

class cVAESolver(Solver):
    """Implements a generator and a discriminator step for the conditional latent regressor model.

    See BaseSolver for detailed docstring.
    # TODO: Either make generic somehow to work with sgan too or implement an sgan version.
    """

    def __init__(self, generator, discriminator, optim=torch.optim.Adam, optims_args=None,
                 lambda_kl=1, loss_fns=None, init_params=False):
        super().__init__(generator, discriminator, optim, optims_args, loss_fns, init_params)
        generator.cvae()
        self.lambda_kl = lambda_kl

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

        loss_fn_gan = self.loss_fns['gan']()
        loss_fn_norm = self.loss_fns['norm']()
        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end, xy_out)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        scores_fake = discriminator(xy_fake, seq_start_end)
        target_fake = torch.ones_like(scores_fake).type(dtype) * random.uniform(0.7, 1.2)

        norm_loss = loss_fn_norm(dxdy_pred, dxdy_out)
        gan_loss = loss_fn_gan(scores_fake, target_fake)
        kl_loss = loss_fn_kl(generator.mu, generator.logvar)

        loss = gan_loss + norm_loss + kl_loss * self.lambda_kl
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

        return gan_loss.item(), norm_loss.item(), loss.item()

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
        loss_fn_gan = self.loss_fns['gan']()

        dxdy_pred = generator(xy_in, dxdy_in, seq_start_end, xy_out)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])

        xy_fake = torch.cat([xy_in, xy_pred], dim=0)
        xy_real = torch.cat([xy_in, xy_out], dim=0)

        scores_fake = discriminator(xy_fake, seq_start_end)
        scores_real = discriminator(xy_real, seq_start_end)

        target_real = torch.ones_like(scores_real).type(dtype) * random.uniform(0.7, 1.2)
        target_fake = torch.zeros_like(scores_fake).type(dtype) * random.uniform(0., 0.3)

        gan_loss_real = loss_fn_gan(scores_real, target_real)
        gan_loss_fake = loss_fn_gan(scores_fake, target_fake)

        loss = gan_loss_fake + gan_loss_real
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        return gan_loss_fake.item(), gan_loss_real.item(), loss.item()



class BicycleSolver(BaseSolver):
    NotImplemented
