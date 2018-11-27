"""
Architechtural modules of social physical attentive multimodal conditional generative adversarial network for tajectory prediction

authors: Janne Lappalainen (email)
         Yagmur Yener (yagmur.yener@tum.de)
"""

import torch
import torch.nn as nn


# Feature Extractors -------------------------------------------------------------------------

class SocialEncoder(nn.Module):
    
    def __init__(input_dim=2, mpl_dim=16, hidden_dim=16, num_layers=1):
        """
        Args:
            input_dim: number of input features (two features: x and y coordinates)
            mlp_dim: spatial embedding dimension of xy coordinates to high dim
            hidden_dim: number of dimensions of the hidden state and therefore the output of LSTM
            num_layers: number of different LSTMs (different Weight sets for each LSTM)
            
        TODO: calc relative distance sorting module
        """
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.mlp_spatial = nn.Sequential(nn.Linear(input_dim, mlp_dim), nn.ReLU())
        self.lstm = nn.Sequential(nn.LSTM(mlp_dim, hidden_dim, num_layers))
        
    def forward(self, input_seq, seq_len=20, batch_size):
        """
        Args:
            input_seq: should be compatible with shape seq_len x batch_dim x input_dim
            seq_len: length of the time sequence
            batch_size: batch size (number of agents ?)
        
        Returns:
            spatial_embedding_seq: xy coordinates embedded in multi dims
            out: all the past hidden states
            hidden: only the last hidden state as output with shape seq_len, batch_size, hidden_size
            encoded_seq: the social property of agent encoded in 64 dim vector (not seq anymore)
            
        TODO: figure out if dimensions fit, also return the relative distance sorting

        """
        spatial_embedding_seq = self.mlp_spatial(input_seq)
        out, hidden = self.lstm(spatial_embedding_seq.view(seq_len, batch_size, -1).float())
        encoded_seq = hidden[0]
        
        #encoded_seq # !!!! change here apply pooling
        
        return encoded_seq
        

        
class PhysicalEncoder(nn.Module)

    def __init__(cnn_type='resnet'):
        '''
        Args:
            cnn_type: a string that tells which pretrained model to use 'resnet' or 'vgg'
        '''
        
        if cnn_type == 'resnet':
            self.cnn = models.resnet18(pretrained=True)
        elif cnn_type == 'vgg'
            self.cnn = models.vgg16(pretrained = True)
            
        modules = list(self.cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        for p in self.cnn.parameters():
            p.requires_grad = False

                
    def forward(input_scene):
        """
        Args:
            Input_scene: input scene at time t or static image ?
        Returns:
            embedded_scene: raw CNN output
        """
        
        embedded_scene = self.cnn(input_scene)
        return embedded_scene
        
        

# Attention Module ---------------------------------------------------------------------------

class SoftAttention(nn.Module):
    '''
    The soft attention module that is used both in the social and physical attention modules after the embedding of features
    '''
    
    def __init__():
        
    
    
    
    
class SocialAttention(nn.Module):
    
    def __init__(social_feats_dim =16, decoder_h_dim=32, mlp_dims=[64,128,64,1], num_max_agents=32):
        """
        Args:
            social_feats_dim: dimension of embedded and sorted social features of input
            decoder_h_dim: dimensions of the hidden state of the decoder
            mlp_dims: dimensions of layers of mlp for embedding social features and hiden states of decoder
            num_max_agents: maximum possible number of agents in a scene
            
        TODO: figure out how num_max_agents are related to this function (set zero non agents)
        """
        
        self.social_feats_dim = social_feats_dim
        self.decoder_h_dim = decoder_h_dim
        self.mlp_dims = mlp_dims
        self.num_max_agents = num_max_agents
        
        self.input_dim = self.decoder_h_dim + self.social_feats_dim 
        self.num_layers = len(self.mlp_dims)
        
        # Embedding
        self.layer1 = nn.Sequential(nn.Linear(self.input_dim, mlp_dims[0]), nn.ReLU)
        self.layer2 = nn.Sequential(nn.Linear(mlp_dims[0]   , mlp_dims[1]), nn.ReLU)
        self.layer3 = nn.Sequential(nn.Linear(mlp_dims[1]   , mlp_dims[2]), nn.ReLU)
        self.layer4 = nn.Sequential(nn.Linear(mlp_dims[2]   , mlp_dims[3]), nn.Softmax) 
        
        # Attention Module
        
    def forward(self, h_states, social_features):
        """
        Args:
            h_states: hidden states of the decoder LSTM in GAN for agents
            social_features: output of encoder, joint social feature vector
        Returns:
            x : the social context. highlights which agents are most important.
        
        TODO: figure out h_states dimensions and fix the concat
        """
        social_embedding = torch.concat(h_states, social_features, dim = 0) # fix here
        social_embedding = self.layer1(x)
        social_embedding = self.layer2(social_embedding)
        social_embedding = self.layer3(social_embedding)
        social_embedding = self.layer4(social_embedding)
        
        return social_embedding
        
    
class PhysicalAttention(nn.Module):
    
    def __init__(in_channels = 512, feat_dim=14, output_dim = 16):
        """
        Args:
            in_channels: number of channels of the raw CNN output images
            feat_dim: image dimensions of the CNN output images
            output_dim: expected output dimension of the fully connected layer  
        """
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        
        # Embedding
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 1, 3), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear((self.feat_dim-2)**2, 16), nn.Softmax())
        
        # Attention Module
        
    def forward(self, h_states, physical_features):
        """
        Args:
          h_states: hidden states of the decoder LSTM in GAN for agents
          physical_features: raw output of VGG or ResNet
          
        Returns:
            x: the physical context. feasible paths. attention weights?
            
        TODO: VGG or ResNet dimensions, what do we do with the attention weights ?
        """
        
        physical_embedding = self.conv(physical_features) 
        physical_embedding = self.mlp(physical_embedding)
        
        return physical_embedding
    
# GAN Modules ------------------------------------------------------------------------------------


class DecoderGAN(nn.Modules): #also called "Generator"
    
    def __init__():
        
        
        
        
class DiscriminatorGAN(nn.Modules):
    
    def __init__():






# Together -------------------------------------------------------------------------------------

class Generator(nn.Modules)

class TrajectoryPredictor(nn.Modules)

















    