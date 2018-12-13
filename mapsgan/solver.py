import random
import torch
from torch import nn
import numpy as np
from mapsgan.utils import get_dtypes, relative_to_abs, init_weights
from mapsgan.losses import l2_loss as loss_fn_l2

long_dtype, dtype = get_dtypes()  # dtype is either torch.FloatTensor or torch.cuda.FloatTensor
cuda = torch.cuda.is_available()


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
        models = [self.generator, self.discriminator]
        if init_params:
            [model.apply(init_weights) for model in models]
        if cuda:
            [model.cuda() for model in models]
        self.optim = optim
        self.optims_args = optims_args
        if optims_args:
            self.optims_args = optims_args
        else:
            self.optims_args = {'generator': {'lr': 1e-3}, 'discriminator': {'lr': 1e-3}}  # default
        if loss_fns:
            self.loss_fns = loss_fns
        else:
            self.loss_fns = {'norm': nn.L1Loss, 'gan': nn.BCEWithLogitsLoss}  # default
        self._reset_histories()

    def train(self, loader, epochs, checkpoint_every=1, steps={'generator': 1, 'discriminator': 1}):
        """Trains the GAN.

        Args:
            loader: Dataloader.
            epochs (int): Number of epochs.
            checkpoint_every (int): Determines for which epochs a checkpoint is created.
            steps (dict): Steps to take on the same batch for generator and discriminator.

        Important: Keep generic to suit all Solvers.
        """
        generator = self.generator
        generator.train()
        discriminator = self.discriminator
        discriminator.train()
        optimizer_g = self.optim(generator.parameters(), **self.optims_args['generator'])
        optimizer_d = self.optim(discriminator.parameters(), **self.optims_args['discriminator'])

        while epochs:
            gsteps = steps['generator']
            dsteps = steps['discriminator']
            for batch in loader:

                while dsteps:
                    losses_d = self.discriminator_step(batch, generator, discriminator, optimizer_d)
                    dsteps -= 1
                while gsteps:
                    losses_g = self.generator_step(batch, generator, discriminator, optimizer_g)
                    gsteps -= 1

            if epochs % checkpoint_every == 0:
                self._checkpoint(losses_g, losses_d)
                self._pprint()

            epochs -= 1

    def test(self, loader, return_gt=False):
        """Tests the generator on unseen data.

        Args:
            loader: Dataloader.
            return_gt: If True, return tuple of (groundtruth, prediction) for each scene instead of (input, prediction).
        """
        generator = self.generator
        generator.eval()
        out = []
        for batch in loader:
            xy_in = batch['xy_in']
            xy_out = batch['xy_out']
            dxdy_in = batch['dxdy_in']
            seq_start_end = batch['seq_start_end']
            dxdy_pred = generator(xy_in, dxdy_in, seq_start_end)
            xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
            for seq in seq_start_end:
                start, end = seq
                if return_gt:
                    out.append((xy_out[:, start:end].numpy(),
                                xy_pred[:, start:end].detach().numpy()))
                else:
                    out.append((xy_in[:, start:end].numpy(),
                                xy_pred[:, start:end].detach().numpy()))
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

    def _pprint(self):
        """Pretty prints the losses."""
        msg = f''
        for type, loss in self.train_loss_history['generator'].items():
            msg += f'{type}: {loss[-1]:.3f}\t'
        for type, loss in self.train_loss_history['discriminator'].items():
            msg += f'{type}: {loss[-1]:.3f}\t'
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
