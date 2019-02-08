from pathlib import Path
import os

import numpy as np
import cv2

root_path = Path(os.path.realpath(__file__)).parent.parent.parent


class Experiment:
    """The experiment objects store mainly paths to train and testfiles as well as homography matrices"""
    def __init__(self):
        super(Experiment, self).__init__()
        self.data_path = ''
        self.video_file = ''
        self.trajectory_file = ''
        self.static_image_file = ''
        self.test_dir = ''
        self.train_dir = ''
        self.val_dir = ''
        self.H = []

    def init_default_args(self):
        self.in_len=8
        self.out_len=12
        self.weight_l2_loss = 1
        self.clip = 1.5
        self.best_k = 10

class ETH(Experiment):

    def __init__(self):
        super().__init__()
        self.data_path = root_path / 'data/eth/'
        self.video_file = self.data_path / 'seq_eth.avi'
        self.trajectory_file = self.data_path / 'test/biwi_eth.txt'
        self.static_image_file = self.data_path / 'eth_static.jpg'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'
        self.H = np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                           [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                           [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]])


class Hotel(Experiment):

    def __init__(self):
        super().__init__()
        self.data_path = root_path / 'data/hotel/'
        self.video_file = self.data_path / 'seq_hotel.avi'
        self.trajectory_file = self.data_path / 'test/biwi_hotel.txt'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'
        self.H = np.array([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                           [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                           [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]])


class Univ(Experiment):

    def __init__(self):
        super(Univ, self).__init__()
        self.data_path = root_path / 'data/univ/'
        self.video_file = self.data_path / 'students003.avi'
        self.trajectory_file = self.data_path / 'test/students003.txt'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'
        self.H = np.array([[0.02104651, 0., -10.01813922],
                           [0., 0.02386598, -2.79231966],
                           [0., 0., 1.]])


class Zara1(Experiment):

    def __init__(self):
        super(Univ, self).__init__()
        self.data_path = root_path / 'data/zara1/'
        self.video_file = self.data_path / 'crowds_zara01.avi'
        self.trajectory_file = self.data_path / 'test/crowds_zara01.txt'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'
        # self.H = np.array([[0.02104651, 0., -10.01813922],
        #                    [0., 0.02386598, -2.79231966],
        #                    [0., 0., 1.]]) # TODO: Check if Univ.H works with plots in ImplementationDetails.ipynb


class Zara2(Experiment):

    def __init__(self):
        super(Univ, self).__init__()
        self.data_path = root_path / 'data/zara3/'
        self.video_file = self.data_path / 'crowds_zara03.avi'
        self.trajectory_file = self.data_path / 'test/crowds_zara03.txt'
        self.test_dir = self.data_path / 'test'
        self.train_dir = self.data_path / 'train'
        self.val_dir = self.data_path / 'val'
        # self.H = np.array([[0.02104651, 0., -10.01813922],
        #                    [0., 0.02386598, -2.79231966],
        #                    [0., 0., 1.]]) # TODO: Check if Univ.H works with plots in ImplementationDetails.ipynb


class Videodata:

    def __init__(self, experiment):
        self.homography = experiment.H
        self.video = cv2.VideoCapture(str(experiment.video_file))
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

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
        return cv2.perspectiveTransform(coords, np.linalg.inv(self.homography)).squeeze()[::-1]

    def getFrame(self, fid):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, fid)
        return self.video.read()[1]

    def staticImage(self):
        ret = True
        image = np.zeros((self.frame_height, self.frame_width, 3))
        while (ret):
            ret, img = self.video.read()
            if not ret:
                break
            image += img
        image /= self.frame_count
        image = image.astype('uint8')
        return image