#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:00:01 2018

@author: j.lappalainen
"""
import os

try:
    torch
except:
    import torch
    from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.distance import pdist, squareform

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
    """Initializes a pytorch dataset. Adaptation from 
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
            train_traj (tensor): Training trajectories #(num_seq, train_len,
                max_peds, 2).
            train_dtraj (tensor): Like train_traj for dx, dy.
            pred_traj (tensor): Like train_traj for the subsequent pred_len
                frames.
            pred_dtraj (tensor): Like pred_traj for dx, dy.
            loss_masks (array): Masks for discriminator of zeros and ones.
            distances (tensor): Pairwise distances of pedestrians. Per sequence
                and timestep. 
    """
            
    
    def __init__(self, data_dir, obs_len = 8, pred_len = 12,
                 min_peds=1, max_peds = 32, delim = '\t'):
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
            data = read_file(path, delim) # (frames, pid, x, y)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[np.where(data[:, 0]==x)] for x in frames]
            for idx, frame in enumerate(frame_data):
                
                # [[idx, 1., 5.4, 8.33], ... , [idx+seq_len, 3., 5.3, 2.9]]
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                
                curr_seq = np.zeros((self.seq_len, self.max_peds, 2)) # (num_peds, xy, seq_len)
                curr_seq_rel = np.zeros((self.seq_len, self.max_peds, 2))
                curr_loss_mask = np.zeros((self.seq_len, self.max_peds))
                peds_considered = 0 # initialize by 0
                
                for _, ped_id in enumerate(peds_in_curr_seq):
                    
                    xy = curr_seq_data[curr_seq_data[:, 1] == ped_id, :] # (frames, ped_id, x, y)
                    
                    #discard pedestrians that are less then seq_len in the whole data
                    xy = np.around(xy, decimals=4)
                    pad_front = frames.index(xy[0, 0]) - idx
                    pad_end = frames.index(xy[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    
                    #import pdb; pdb.set_trace();
                    # Get dx and dy (relative coordinates)
                    xy = xy[:, 2:] # (2, seq_len)
                    dxy = np.zeros(xy.shape)
                    dxy[1:, :] = xy[1:, :] - xy[:-1, :] # dx, dy
                    _idx = peds_considered
                    curr_seq[pad_front:pad_end,  _idx, :] = xy # fill array with all pedestrians
                    curr_seq_rel[pad_front:pad_end, _idx, :] = dxy
                    curr_loss_mask[pad_front:pad_end, _idx] = 1 # ground truth for discriminator ?
                    peds_considered += 1
                
                # append to data if more than min_peds were observed
                if peds_considered > self.min_peds:
                    self.peds_per_seq.append(peds_considered)
                    self.loss_masks.append(curr_loss_mask)
                    trajectories.append(curr_seq) # (num_seq, num_peds, xy, seq_len)
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
        self.train_traj = torch.Tensor(trajectories[:, :self.train_len]).float()
        self.train_dtraj = torch.Tensor(dtrajectories[:, :self.train_len]).float()
        self.pred_traj = torch.Tensor(trajectories[:, :self.pred_len]).float()
        self.pred_dtraj = torch.Tensor(dtrajectories[:, :self.pred_len]).float()
        self.loss_masks = torch.Tensor(self.loss_masks).float()     
        self.distances = torch.Tensor(self.distances).float()

    def __len__(self):
        return len(self.train_traj)
    
    def __getitem__(self, idx):
        out = {'train': self.train_traj[idx],
               'dtrain': self.train_dtraj[idx],
               'groundtruth': self.pred_traj[idx],
               'dgroundtruth': self.pred_dtraj[idx],
               'distances': self.distances[idx],
               'lossmask': self.loss_masks[idx]}
        return out