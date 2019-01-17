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
    dtype = get_dtypes()[1]
    if noise_type == 'gaussian':
        return torch.randn(*shape).type(dtype)
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).type(dtype)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_z_random(batch_size, z_dim, random_type='gauss'):
    dtype = get_dtypes()[1]
    if random_type == 'uni':
        z = torch.rand(batch_size, z_dim) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, z_dim)
    return z.type(dtype)


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


def norm_sequence(seq):
    """Normalizes a seq of shape (seq_len, num_agents, num_coords)
    per trajectory.
    """
    eps = 1e-10
    normed = np.zeros_like(seq)
    seq = seq.transpose((1, 0, 2))
    for i, s in enumerate(seq):
        normed[:, i, :] = (s - s.min(axis=0)) / (s.max(axis=0) - s.min(axis=0) + eps)
    return normed


def norm_scene(scene):
    """Normalize all sequences within a scene.

    Args:
        scene (list): List of sequences of shape expected by norm_sequence.
    """
    normed = []
    for seq in scene:
        normed.append(norm_sequence(seq))
    return normed


def cos_sequence(seq):
    """Computes the cosine distance of seq of shape (seq_len, num_agents, num_coords)
    per trajectory.
    """
    cos = nn.CosineSimilarity(dim=0)
    eps = 1e-10
    num_agents = seq.shape[1]
    distance = torch.zeros([num_agents, num_agents])
    seq = seq.transpose(1, 0)
    ind = np.triu_indices(num_agents, k=1)
    for i, s1 in enumerate(seq):
        for j, s2 in enumerate(seq):
            distance[i, j] = 1 - cos(s1.flatten(), s2.flatten())
    return distance  # [ind]


def cos_scene(scene):
    """Summed cosine distance for all sequences within a scene.

    Args:
        scene (list): List of sequences of shape expected by cos_sequence.
    """
    distances = []
    for seq in scene:
        seq = torch.Tensor(seq)
        distances.append(cos_sequence(seq).sum())
    return sum(distances)