import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class Trajectories(Dataset):
    """Initializes a pytorch dataset. Adaptation of
    https://github.com/agrimgupta92/sgan/blob/master/sgan/data/trajectories.py.
    
    Args:
        data_dir (str): Path towards folder with trajectory files.
        delim (str): Delimiter for reading the trajectory files.
        train_len (int): Length of training sequence.
        pred_len (int): Length of groundtruth sequence.
        min_peds (int): Minimal number of pedestrians.
        max_peds (int): Maximal number of pedestrians.

    Attributes:
        data_dir (str): Path towards folder with trajectory files.
        all_files (list): List of all files inside data_dir.
        train_len (int): Length of training sequence.
        pred_len (int): Length of groundtruth sequence.
        min_peds (int): Minimal number of pedestrians.
        max_peds (int): Maximal number of pedestrians.
        seq_len (int): Training sequence length + groundtruth length.
        peds_per_seq (list): Number of agents per sequence.
        xy_in (tensor): Training trajectories #(num_seq, train_len,
            max_peds, 2).
        dxdy_in (tensor): Like xy_in for dx, dy.
        xy_out (tensor): Like xy_in for the subsequent pred_len
            frames.
        dxdy_out (tensor): Like xy_out for dx, dy.
        loss_masks (array): Masks for discriminator of zeros and ones.
        distances (tensor): Pairwise distances of pedestrians. Per sequence
            and timestep.
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, min_peds=1, max_peds=32, delim='\t'):
        super(Trajectories, self).__init__()
        self.data_dir = data_dir
        all_files = os.listdir(self.data_dir)
        self.all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        self.min_peds = min_peds
        self.max_peds = max_peds
        self.train_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.train_len + self.pred_len

        self.peds_per_seq = []
        self.loss_masks = []

        trajectories = []
        dtrajectories = []

        for path in self.all_files:
            data = read_file(path, delim)  # (frames, pid, x, y)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[np.where(data[:, 0] == x)] for x in frames]
            for idx, frame in enumerate(frame_data):

                # [[idx, 1., 5.4, 8.33], ... , [idx+seq_len, 3., 5.3, 2.9]]
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                curr_seq = np.zeros((self.seq_len, self.max_peds, 2))  # (num_peds, xy, seq_len)
                curr_seq_rel = np.zeros((self.seq_len, self.max_peds, 2))
                curr_loss_mask = np.zeros((self.seq_len, self.max_peds))
                peds_considered = 0  # initialize by 0

                for _, ped_id in enumerate(peds_in_curr_seq):

                    xy = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]  # (frames, ped_id, x, y)

                    # discard pedestrians that are less then seq_len in the whole data
                    xy = np.around(xy, decimals=4)
                    pad_front = frames.index(xy[0, 0]) - idx
                    pad_end = frames.index(xy[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue

                    # import pdb; pdb.set_trace();
                    # Get dx and dy (relative coordinates)
                    xy = xy[:, 2:]  # (2, seq_len)
                    dxy = np.zeros(xy.shape)
                    dxy[1:, :] = xy[1:, :] - xy[:-1, :]  # dx, dy
                    _idx = peds_considered
                    curr_seq[pad_front:pad_end, _idx, :] = xy  # fill array with all pedestrians
                    curr_seq_rel[pad_front:pad_end, _idx, :] = dxy
                    curr_loss_mask[pad_front:pad_end, _idx] = 1  # ground truth for discriminator ?
                    peds_considered += 1

                # append to data if more than min_peds were observed
                if peds_considered > self.min_peds:
                    self.peds_per_seq.append(peds_considered)
                    self.loss_masks.append(curr_loss_mask)
                    trajectories.append(curr_seq)  # (num_seq, num_peds, xy, seq_len)
                    dtrajectories.append(curr_seq_rel)

        # List -> Array         
        trajectories = np.array(trajectories)
        dtrajectories = np.array(dtrajectories)
        self.loss_masks = np.array(self.loss_masks)

        # Compute pairwise distances
        self.distances = np.zeros([len(trajectories), self.seq_len,
                                   self.max_peds, self.max_peds])
        for traj_idx, traj in enumerate(trajectories):
            for seq_idx, seq in enumerate(traj):
                dist = squareform(pdist(seq), 'euclidean')
                num_peds = self.peds_per_seq[traj_idx]
                _filldist = (dist.max() + 1).round()
                dist[num_peds::] = _filldist
                dist[:, num_peds::] = _filldist
                self.distances[traj_idx, seq_idx] = dist

        # Array -> Torch Tensor
        self.xy_in = torch.Tensor(trajectories[:, :self.train_len]).float()
        self.dxdy_in = torch.Tensor(dtrajectories[:, :self.train_len]).float()
        self.xy_out = torch.Tensor(trajectories[:, :self.pred_len]).float()
        self.dxdy_out = torch.Tensor(dtrajectories[:, :self.pred_len]).float()
        self.loss_masks = torch.Tensor(self.loss_masks).float()
        self.distances = torch.Tensor(self.distances).float()

    def __len__(self):
        return len(self.xy_in)

    def __getitem__(self, idx):
        out = {'xy_in': self.xy_in[idx],
               'dxdy_in': self.dxdy_in[idx],
               'xy_out': self.xy_out[idx],
               'dxdy_out': self.dxdy_out[idx],
               'distances': self.distances[idx],
               'lossmask': self.loss_masks[idx]}
        return out


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets

    Args:
        data_dir: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y>
        obs_len: Number of time-steps in input trajectories
        pred_len: Number of time-steps in output trajectories
        skip: Number of frames to skip while making the dataset
        threshold: Minimum error to be considered for non linear when using a linear predictor
        min_ped:   Minimum number of pedestrians that should be in a seqeunce
        delim:  Delimiter in the dataset files

    TODO: Add distances.
    """
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=1, delim='\t'):

        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    curr_seq = curr_seq[:num_peds_considered]
                    seq_list.append(curr_seq)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list = torch.from_numpy(seq_list).type(torch.float)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = seq_list[:, :, :self.obs_len].contiguous()
        self.pred_traj = seq_list[:, :, self.obs_len:].contiguous()
        displ = displacement(seq_list)
        self.obs_traj_rel = displ[:, :, :self.obs_len].contiguous()
        self.pred_traj_rel = displ[:, :, self.obs_len:].contiguous()
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return (self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                self.non_linear_ped[start:end], self.loss_mask[start:end, :])


def displacement(seq):
    """Calculates displacement as in sgan.

    Args:
        seq (tensor): Tensor of shape (num_agents, num_coords, seq_len)

    Returns:
        tensor: Tensor with displacements of the same shape as seq.
    """
    disp = torch.zeros_like(seq).type(torch.float)
    for i, s in enumerate(seq):
        disp[i, :, 1:] = s[:, 1:] - s[:, :-1]
    return disp


def seq_collate(data):
    """Merges a sample-list of length batchsize (specified in data_loader).

    Args:
        data: List of samples from the dataset. Each sample is a tensor of shape (num_peds (batch_dim), 2, seq_len).

    Returns:
        tuple: Concatenated samples over batchsize. Transposed so that (seq_len, batchsize*num_peds, 2).
    """
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = {'xy_in': obs_traj,
           'dxdy_in': obs_traj_rel,
           'xy_out': pred_traj,
           'dxdy_out': pred_traj_rel,
           'non_linear_ped': non_linear_ped,
           'loss_mask': loss_mask,
           'seq_start_end': seq_start_end}

    return out


def data_loader(in_len, out_len, batch_size, num_workers, path, shuffle=True):
    dset = TrajectoryDataset(path,
                             obs_len=in_len,
                             pred_len=out_len)

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=seq_collate)
    return dset, loader
