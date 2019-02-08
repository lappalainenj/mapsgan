import torch
from torch import nn
import numpy as np

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
    """Get a random vector of shape (batch_size, z_dim)."""
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
    return distance[ind]


def cos_scene(scene, intra=False):
    """Summed cosine distance for all sequences within a scene.
    Note: for within agent cosine score the loop is over agents.

    Args:
        scene (list): List of sequences of shape expected by cos_sequence.
    """
    distances = []
    for seq in scene:
        seq = torch.Tensor(seq)
        distances.append(cos_sequence(seq).sum())
    if intra:
        return np.mean(distances)
    return sum(distances).item()


def get_cosine_score(output):
    """Get the normalized cosine score for a scene output."""
    return cos_scene(output['xy_pred']) / cos_scene(output['xy_out'])


def get_intra_agent_distance(int_output):
    """Get intra agent distance from interpolation."""
    num_steps = int_output['xy_pred'][0].shape[0]
    num_agents = int_output['xy_pred'][0].shape[1]
    num_coord = int_output['xy_pred'][0].shape[2]
    num_z = len(int_output['xy_pred'])
    agent_interp = [None] * num_agents
    agent_array = np.zeros([num_steps, num_z, num_coord])
    for agent in range(num_agents):
        for z in range(num_z):
            agent_array[:, z, :] = int_output['xy_pred'][z][:, agent, :]
        agent_interp[agent] = agent_array
    #agent_interp = norm_scene(agent_interp)
    return cos_scene(agent_interp, intra=True)

def get_average_within_agent_cosine( generator, testloader, scenes=[7, 25, 42, 44, 36]):
    """Average within agent diversity across sampled scenes."""
    from mapsgan import BaseSolver
    solver = BaseSolver(generator, None)
    distance = 0
    for scene in scenes:
        inter = solver.interpolate(testloader, scene=scene, stepsize=0.3)
        distance += get_intra_agent_distance(inter)
    distance /= len(scenes)
    return distance

def get_collisions(output, thresh=0.5):
    """Computes collision on array of (seq_len, num_agents, num_chords)."""
    from scipy.spatial.distance import pdist, squareform
    collisions = 0
    xy_pred = output['xy_pred']
    for scene in xy_pred:
        for i, step_na_xy in enumerate(scene):
            dm = squareform(pdist(step_na_xy))  # step is (num_agents, dimensions xy)
            ind = np.triu_indices(dm.shape[0], k=1)
            for distance in dm[ind]:
                if distance < thresh:
                    collisions += 1
    return collisions


def get_average_fde(output):
    """Get the average displacement error."""
    pred = output['xy_pred']
    gt = output['xy_out']
    diff = 0
    for i, p in enumerate(pred):
        last_p = p[-1]
        last_gt = gt[i][-1]
        diff += np.linalg.norm((last_p-last_gt), axis=1).mean()
    diff /= len(pred)
    return diff


def smooth_data(data, N):
    """
    Running average filtering for smoothing the plotted losses

    Args:
        data (list): one of the accuracy or diversity scores
        N (int): size of the smoothing window in terms of data points

    Returns:
        moving_aves (list): smoothed scores

    """
    cumsum, moving_aves = [0], []
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
    return moving_aves


def get_sgan_generator(checkpoint, cuda=False):
    """Adapted from sgan.evaluation. Get sgan generator from checkpoint."""
    from attrdict import AttrDict
    from mapsgan.sgan import TrajectoryGenerator
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    if cuda:
        generator.cuda()
    generator.train()
    return generator


def get_sgan_bicy(ckpt_bicy):
    """Get sgan that we finetuned in bicycle mode."""
    from collections import OrderedDict
    state_dict = OrderedDict()
    for key in ckpt_bicy['g_state']:
        if 'generator.' in key:
            new_key = key.replace('generator.', '')
            state_dict[new_key] = ckpt_bicy['g_state'][key]
    out = {}
    out['args'] = {'encoder_h_dim_d': 48,
                     'num_layers': 1,
                     'neighborhood_size': 2.0,
                     'pool_every_timestep': False,
                     'clipping_threshold_g': 2.0,
                     'delim': 'tab',
                     'print_every': 100,
                     'skip': 1,
                     'loader_num_workers': 4,
                     'obs_len': 8,
                     'encoder_h_dim_g': 32,
                     'batch_size': 64,
                     'num_epochs': 200,
                     'best_k': 20,
                     'd_steps': 1,
                     'pred_len': 12,
                     'g_steps': 1,
                     'g_learning_rate': 0.0001,
                     'l2_loss_weight': 1.0,
                     'grid_size': 8,
                     'bottleneck_dim': 8,
                     'checkpoint_name': 'checkpoint',
                     'gpu_num': '0',
                     'restore_from_checkpoint': 1,
                     'dropout': 0.0,
                     'checkpoint_every': 300,
                     'noise_mix_type': 'global',
                     'decoder_h_dim_g': 32,
                     'pooling_type': 'pool_net',
                     'use_gpu': 1,
                     'num_iterations': 7818,
                     'batch_norm': False,
                     'noise_type': 'gaussian',
                     'clipping_threshold_d': 0,
                     'd_learning_rate': 0.001,
                     'checkpoint_start_from': None,
                     'timing': 0,
                     'mlp_dim': 64,
                     'num_samples_check': 5000,
                     'd_type': 'global',
                     'noise_dim': (8,),
                     'dataset_name': 'eth',
                     'embedding_dim': 16}
    out['g_state'] = state_dict
    sgan_bicy = get_sgan_generator(out)
    return sgan_bicy


def get_model_dict(folder, model_path = '../models/final/eth', sgan_pretrained_file = '/sgan_eth_12_model.pt'):
    """Get a dictionary of all evaluated models."""
    from mapsgan import ToyGenerator, BicycleGenerator
    import os
    in_len, out_len = 8, 12
    files = os.listdir(folder)
    cuda = torch.cuda.is_available()
    ml = 'cpu' if not cuda else None
    ##
    ckpt = torch.load(model_path + sgan_pretrained_file, map_location=ml)
    ##
    sgan = get_sgan_generator(ckpt)
    models = {'toy':{'generator':ToyGenerator(in_len=in_len, out_len=out_len)},
             'clr':{'generator':BicycleGenerator(ToyGenerator, start_mode='clr')},
             'cvae':{'generator':BicycleGenerator(ToyGenerator, start_mode='cvae')},
             'bicy':{'generator':BicycleGenerator(ToyGenerator, start_mode='cvae')},
             'sgan':{'generator':sgan},
             'sgn_bcy':{'generator':None}}
    model_keys = [key for key in models.keys()]
    for key in model_keys:
        for file in files:
            if key in file:
                checkpoint = torch.load(folder + '/' + file, map_location=ml)
                if key == 'sgan':
                    pass
                elif key == 'sgn_bcy':
                    models[key].update({'generator': get_sgan_bicy(checkpoint)})
                else:
                    models[key]['generator'].load_state_dict(checkpoint["g_state"])
                try:
                    models[key].update({'train_loss_history': checkpoint["train_loss_history"]})
                except:
                    pass
                models[key].update({'ckpt': file})
    return models


def get_metrics(model, testloader):
    """Computes four metrics for given generator and testloader."""
    from mapsgan import BaseSolver
    solver = BaseSolver(None, None)
    metrics = {'collisions':None,
              'fde':None,
              'inter_div':None,
              'intra_div':None}
    solver.generator = model
    output = solver.test(testloader)
    #int_out = solver.interpolate(testloader, stepsize=0.9, scene=44)
    metrics['fde'] = get_average_fde(output)
    metrics['collisions'] = get_collisions(output)
    metrics['inter_div'] = get_cosine_score(output)
    metrics['intra_div'] = get_average_within_agent_cosine(model, testloader)#
    return metrics


def get_metrics_all(models, experiment):
    """Computes metrics for all generator stored in models."""
    from mapsgan import data_loader
    _, testloader = data_loader(in_len=8,
                                   out_len=12,
                                   batch_size=1,
                                   num_workers=1,
                                   path=experiment.test_dir,
                                   shuffle=False)
    model_metrics = {key: None for key in models.keys()}
    for key in model_metrics:
        if models[key]['generator']:
            model_metrics[key] = get_metrics(models[key]['generator'], testloader)
    return model_metrics