import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

#root_path = Path('/mnt/Clouds/MultimodalSophie/')
root_path = Path('/home/yy/ADL4CV/mapsgan/')


class Experiment:
    
    def __init__(self):
        super(Experiment, self).__init__()
        self.file_name = ''
        self.file_path = ''
        
class ETH(Experiment):

    def __init__(self):
        super(ETH, self).__init__()
        self.file_name = 'seq_eth.avi'
        self.file_path = root_path / 'data/ETH/ewap_dataset/seq_eth'
        self.video_file = root_path / 'data/ETH/ewap_dataset/seq_eth/seq_eth.avi'
        self.H = np.array([[2.8128700e-02,   2.0091900e-03,  -4.6693600e+00], 
           [8.0625700e-04,   2.5195500e-02,  -5.0608800e+00],
           [3.4555400e-04,   9.2512200e-05,   4.6255300e-01]])

class Videodata:
    
    def __init__(self, experiment):
        self.file_path = experiment.file_path
        self.file_name = experiment.file_name
        self.homography = experiment.H
        self.video = cv2.VideoCapture(str(experiment.video_file))
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __getitem__(self, key):
        return NotImplemented
    
    def __len__(self):
        return NotImplemented
    
    def read_file(self, _path, delim='\t'):
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
    
    def camcoordinates(self, xy):
        """Transform the meter coordinates with the homography matrix"""
        coords = xy.reshape(1, 1, -1)
        return cv2.perspectiveTransform(coords,  np.linalg.inv(self.homography)).squeeze()[::-1]
    
    def getFrame(self, fid):
        self.video.set(cv2.CAP_PROP_POS_FRAMES,fid)
        return self.video.read()[1]
    
    def staticImage(self):
        ret = True
        image = np.zeros((self.frame_height, self.frame_width, 3))
        while(ret):
            ret, img = self.video.read()
            if not ret:
                break
            image += img
        image /= self.frame_count
        image = image.astype('uint8')
        return image
        
    