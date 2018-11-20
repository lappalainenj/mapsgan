"""
Architechtural modules of social physical attentive multimodal conditional generative adversarial network for tajectory prediction

authors: Janne Lappalainen (email)
         Yagmur Yener (yagmur.yener@tum.de)
"""

import torch
import torch.nn as nn

"""In the social attention module, attention weights are
retrieved by passing the encoder output and decoder context
through multiple MLP layers of sizes 64, 128, 64, and 1,
with interspersed ReLu activations. The final layer is passed
through a Softmax layer. The interactions of the surrounding
Nmax = 32 agents are considered; this value was chosen
as no scenes in either dataset exceeded this number of total
active agents in any given timestep. If there are less than
Nmax agents, the dummy value of 0 is used. 



The physical
attention module takes raw VGG features (512 channels),
projects those using a convolutional layer, and embeds those
using a single MLP to an embedding dimension of 16.

"""

class SocialAttention(nn.Module):
    
    def __init__(num_max_agents=32, num_social_feats=16, mlp_dims=[64,128,64,1]):
        
        self.num_max_agents = num_max_agents
        self.num_social_feats = num_social_feats
        self.mlp_dims=mlp_dims
        
        self.input_dim = self.num_max_agents + self.num_social_feats
        self.num_layers=len(self.mlp_dims)
        
        self.layer1 = nn.Sequential(nn.Linear(self.input_dim, mlp_dims[0]), nn.ReLU)
        self.layer2 = nn.Sequential(nn.Linear(mlp_dims[0]   , mlp_dims[1]), nn.ReLU)
        self.layer3 = nn.Sequential(nn.Linear(mlp_dims[1]   , mlp_dims[2]), nn.ReLU)
        self.layer4 = nn.Sequential(nn.Linear(mlp_dims[2]   , mlp_dims[3]), nn.ReLU, nn.Softmax)
        
    def forward(self, h_states, social_features):
        """
        Args:
            h_states: hidden states of the decoder LSTM in GAN for agents
            social_features: output of encoder, joint social feature vector
        Returns:
            x : the social context. highlights which agents are most important.
        
        TODO: figure out h_states dimensions and fix the concat
        """
        x = torch.concat(h_states, social_features, dim = 0) # fix here
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
        
    
class PhysicalAttention(nn.Module):
    
    def __init__(in_channels = 512, feat_dim=14, output_dim = 16):
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, 1, 3), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(self.feat_dim-2, 16), nn.Softmax())
        
    def forward(self, h_states, physical_features):
        """
        Args:
          h_states: hidden states of the decoder LSTM in GAN for agents
          physical_features: output of VGG or ResNet
        Returns:
            x: the physical context. feasible paths
            
        TODO: h_states and VGG or ResNet dimensions
        """
        
        x = torch.concat(h_states, physical_features, dim = 0) # fix here
        x = self.conv(x)
        x = self.mlp(x)
        
        return x
                
    