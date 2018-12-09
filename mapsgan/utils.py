import torch
from torch import nn

def relative_to_abs(rel_traj, start_pos):
    """Given the initial positions, computes the absolute trajectory from displacements.

    Args:
        rel_traj: Tensor of shape (seq_len, batch, 2). Displacements.
        start_pos: Tensor of shape (batch, 2). Initial positions.

    Returns:
        Tensor of shape (seq_len, batch, 2). Absolute trajectories.
    """
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def get_dtypes():
    """Returns either cuda-dtype or cpu-dtype of sort long and float.

    Args:
        args: Parsed arguments. Only args.use_gpu is required here.

    Returns:
        long_dtype
        float_dtype

    """
    if torch.cuda.is_available():
        return torch.cuda.LongTensor, torch.cuda.FloatTensor
    return torch.LongTensor, torch.FloatTensor


def get_noise(shape, noise_type):
    """Create a noise vector.

    Args:
        shape: Shape of the vector.
        noise_type: 'gaussian' or 'uniform' between -1, 1.

    Returns:
        tensor

    """
    if noise_type == 'gaussian':
        return torch.randn(*shape).type(dtype)
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).type(dtype)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0.):
    """Create a multilayer perceptron.

    Args:
        dim_list: List of subsequent dimensions. Determines number of layers.
        activation: 'relu' or 'leakyrelu' as activation functions.
        batch_norm: Add batch normalization after? the linear layer.
        dropout: Add dropout after the activation.

    Returns:
        nn.Sequential(*layers)

    """
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def init_weights(m):
    """Initializes Linear layers to Kiaming Normal."""
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)