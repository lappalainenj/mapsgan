import torch
import random

bce_loss = torch.nn.BCEWithLogitsLoss()

def gan_g_loss(scores_fake):
    """Computes the BCE Loss for the generator, given the scores from the Discriminator.

    Args:
        scores_fake: Tensor of shape (N,) containing scores for fake samples

    Returns:
        Tensor of shape (,) giving GAN generator loss

    Note: adds a random scalar to the scores as regularization.
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2) # IMPORTANT: Ones because G needs to trick D
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """Computes the BCE Loss for the discriminator, given its scores real and fake data.

    Args:
        scores_real: Tensor of shape (N,) giving scores for real samples.
        scores_fake: Tensor of shape (N,) giving scores for fake samples.

    Returns:
        Tensor of shape (,) giving GAN discriminator loss.

    Note: adds a random scalar to the scores as regularization.
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2) # IMPORTANT: Ones because D needs to learn truth
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3) # IMPORTANT: Zeros because D needs to learn fake
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    """Computes the L2Loss between predicted and groundtruth trajectories.

    Args:
        pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
        pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth.
        loss_mask: Tensor of shape (batch, seq_len)
        mode: Either sum, average, or raw.
    Returns:
        Tensor of shape (,) giving l2 loss.
    """
    seq_len, batch, _ = pred_traj.size()
    # QUESTION: Why exactly multiply with loss_mask? Zero out non-existing agents?
    loss = (loss_mask.unsqueeze(dim=2) * (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """Computes the displacement error between predicted and groundtruth trajectories.

    Args:
        pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
        pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth.
        consider_ped: Tensor of shape (batch).
        mode: Either sum or raw.

    Returns:
        Tensor of shape (,) giving the eculidian displacement error.

    # QUESTION: How is this different from sqrt(l2loss)?
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """Computes the displacement error between predicted and groundtruth destinations.

    Args:
        pred_pos: Tensor of shape (batch, 2). Predicted last positions.
        pred_pos_gt: Tensor of shape (seq_len, batch, 2). True last positions.
        consider_ped: Tensor of shape (batch).
        mode: Either sum or raw.

    Returns:
        Tensor of shape (,) giving the eculidian displacement error.
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
