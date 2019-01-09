import torch
import torch.nn as nn

from mapsgan.utils import get_dtypes, make_mlp, get_noise, get_z_random, relative_to_abs

long_dtype, dtype = get_dtypes()  # dtype is either torch.FloatTensor or torch.cuda.FloatTensor


class Encoder(nn.Module):
    """Encoder, part of both Generator and Discriminator.

        Args:
            embedding_dim (int): Output dim of embedding (2 -> embedding_dim, via Linear layer).
            h_dim (int): Hidden dim of the LSTM (embedding_dim -> h_dim).
            num_layers (int): Number of stacked lstms.
            dropout (float): Specifies dropout in the lstm layer.

        Attributes:
            embedding_dim (int): Output dim of linear layer: 2 -> embedding_dim.
            h_dim (int): Hidden dim of the LSTM: embedding_dim -> h_dim.
            num_layers (int): Number of stacked lstms.
            embedding (nn.Linear): Embeds x and y coordinates into embedding_dim dimensions.
            encoder (nn.LSTM): Encodes trajectory information.

    """

    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.h_dim).type(dtype),
                torch.zeros(self.num_layers, batch, self.h_dim).type(dtype))

    def forward(self, xy_in):
        """ Forward function of the Trajectory Encoder.

            Args:
                xy_in (tensor): Tensor of shape (in_len, batch_size, 2).

            Returns:
                tensor: Tensor of shape (self.num_layers, batch, self.h_dim).

            Note: The batchsize does not need to be static.
        """
        # Encode observed Trajectory
        batch = xy_in.size(1)
        xy_in_embedding = self.embedding(xy_in.contiguous().view(-1, 2))
        xy_in_embedding = xy_in_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(xy_in_embedding, state_tuple)
        hidden = state[0]
        return hidden


class ToyDecoder(nn.Module):
    """Decoder, part of Generator that predicts displacements.

        Args:
            out_len (int): Length of the sequence to predict.
            embedding_dim (int): Output dim of embedding (2 -> embedding_dim, via Linear layer). Defaults to 64.
            h_dim (int): Hidden dim of the LSTM (embedding_dim -> h_dim -> 2). Defaults to 128.
            num_layers (int): Number of stacked lstms. Defaults to 1.
            dropout (float): Dropout for LSTM and pooling. Defaults to 0.0.

        Attributes:
            out_len (int): Length of the sequence.
            embedding_dim (int): Output dim of embedding (2 -> embedding_dim, via Linear layer).
            h_dim (int): Hidden dim of the LSTM (embedding_dim -> h_dim -> 2).
            embedding (nn.Linear): Embedding layer.
            decoder (nn.LSTM): Single sequence length decoding lstm.
            hidden2pos (nn.Linear): Linear layer, transforming LSTMs hidden to coordinates.
    """

    def __init__(self, out_len, embedding_dim=64, h_dim=128, num_layers=1, dropout=0.0):
        super(ToyDecoder, self).__init__()

        self.out_len = out_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim

        self.embedding = nn.Linear(2, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, xy_last, dxdy_last, state_tuple, seq_start_end):
        """Forward function of the decoder.

        Args:
            xy_last (tensor): Last positions of shape (batch, 2). # Important: Only last.
            dxdy_last (tensor): Last displacements of shape (batch, 2).
            state_tuple (tensor): Hidden state of Generator (hh, ch). Each tensor of shape (num_layers, batch, h_dim).
            seq_start_end: A list of tuples which delimit sequences within batch.

        Returns:
            tensor: tensor of shape (self.out_len, batch, 2).
        """
        batch = xy_last.size(0)
        dxdy_pred = []
        decoder_input = self.embedding(dxdy_last)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.out_len):
            # Important: Predict next displacements, given state tuple and embedded displacements.

            output, state_tuple = self.decoder(decoder_input, state_tuple)
            dxdy_next = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = dxdy_next + xy_last
            decoder_input = self.embedding(dxdy_next)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            dxdy_pred.append(dxdy_next.view(batch, -1))
            xy_last = curr_pos

        dxdy_pred = torch.stack(dxdy_pred, dim=0)
        return dxdy_pred, state_tuple[0]


class ToyGenerator(nn.Module):
    """ Generator, combining decoder, encoder and pooling.
    Args:
        in_len (int):
        out_len (int):
        embedding_dim (int):
        encoder_h_dim (int):
        decoder_h_dim (int):
        mlp_dim (int):
        num_layers (int):
        noise_dim (tuple):
        noise_type (str):
        noise_mix_type (str): Either 'ped' or 'global'. If 'global', noise is replicated along ped dimension. If 'ped',
            all get independent noise.
        dropout (float):
        activation (str):
        batch_norm (bool):
    """

    def __init__(self, in_len, out_len, embedding_dim=64, encoder_h_dim=64, z_dim=8,
                 num_layers=1, noise_type='gaussian', noise_mix_type='ped', dropout=0.0, **kwargs):

        super().__init__()

        self.in_len = in_len
        self.out_len = out_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.z_dim = z_dim
        decoder_h_dim = encoder_h_dim + z_dim
        self.decoder_h_dim = decoder_h_dim
        self.noise_dim = (z_dim,)
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type

        self.encoder = Encoder(embedding_dim=embedding_dim,
                               h_dim=encoder_h_dim,
                               num_layers=num_layers,
                               dropout=dropout)

        self.decoder = ToyDecoder(out_len,
                                  embedding_dim=embedding_dim,
                                  h_dim=decoder_h_dim,
                                  num_layers=num_layers,
                                  dropout=dropout, )

    def forward(self, xy_in, dxdy_in, seq_start_end, user_noise=None):
        """Forward function of TrajectoryGenerator.

        Args:
            xy_in: Tensor of shape (in_len, batch, 2).
            dxdy_in: Tensor of shape (in_len, batch, 2).
            seq_start_end: A list of tuples which delimit sequences within batch.
            user_noise: Generally used for inference when you want to see
                relation between different types of noise and outputs.

        Returns:
            Tensor of shape (self.out_len, batch, 2)
        """
        batch = dxdy_in.size(1)
        # Encode seq
        encoded = self.encoder(dxdy_in).view(-1, self.encoder_h_dim)

        # Add noise
        decoder_h = self.add_noise(encoded, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).type(dtype)
        state_tuple = (decoder_h, decoder_c)

        xy_last = xy_in[-1]
        dxdy_last = dxdy_in[-1]
        # Predict Trajectory
        decoder_out = self.decoder(xy_last, dxdy_last, state_tuple, seq_start_end)
        dxdy_pred, _ = decoder_out

        return dxdy_pred

    def add_noise(self, encoded, seq_start_end, user_noise=None):
        """Concatenates the input vector with a noise vector.

        Args:
            encoded: Tensor of shape (_, decoder_h_dim - noise_first_dim).
            seq_start_end: A list of tuples which delimit sequences within batch.
            user_noise: Generally used for inference when you want to see
                relation between different types of noise and outputs.

        Returns:
            Tensor of shape (_, decoder_h_dim)
        """
        if not self.z_dim:
            return encoded

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (encoded.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([encoded[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([encoded, z_decoder], dim=1)

        return decoder_h


class ToyDiscriminator(nn.Module):
    """

    Args:
        embedding_dim:
        h_dim:
        mlp_dim:
        num_layers:
        activation:
        batch_norm:
        dropout:
        d_type:
    """

    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
                 activation='relu', batch_norm=True, dropout=0.0):
        super(ToyDiscriminator, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim

        self.encoder = Encoder(embedding_dim=embedding_dim,
                               h_dim=h_dim,
                               num_layers=num_layers,
                               dropout=dropout)

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims,
                                        activation=activation,
                                        batch_norm=batch_norm,
                                        dropout=dropout)

    def forward(self, xy, seq_start_end=None):
        """
        Args:
            xy: Tensor of shape (in_len + out_len, batch, 2).
            seq_start_end: A list of tuples which delimit sequences within batch.

        Returns:
            Tensor of shape (batch,) with real/fake scores.
        """
        hidden = self.encoder(xy)
        classifier_input = hidden.squeeze()
        scores = self.real_classifier(classifier_input)
        return scores


class BicycleEncoder(nn.Module):
    """BicycleEncoder.

        Args:
            embedding_dim (int): Output dim of embedding (2 -> embedding_dim, via Linear layer).
            h_dim (int): Hidden dim of the LSTM (embedding_dim -> h_dim).
            z_dim (int): Dimension of noise input.
            num_layers (int): Number of stacked lstms.
            dropout (float): Specifies dropout in the lstm layer.

        Attributes:
            embedding_dim (int): Output dim of linear layer: 2 -> embedding_dim.
            h_dim (int): Hidden dim of the LSTM: embedding_dim -> h_dim.
            num_layers (int): Number of stacked lstms.
            embedding (nn.Linear): Embeds x and y coordinates into embedding_dim dimensions.
            encoder (nn.LSTM): Encodes trajectory information.
            fc (nn.Linear): Projects encoding to mean estimates.
            fclogvar (nn.Linear): Projects encoding to logvar=2*logstd estimates.

    """

    def __init__(self, embedding_dim=64, h_dim=64, z_dim=8, num_layers=1, dropout=0.0):
        super(BicycleEncoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.fc = nn.Linear(h_dim * num_layers, z_dim)
        self.fclogvar = nn.Linear(h_dim * num_layers, z_dim)

    def init_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.h_dim).type(dtype),
                torch.zeros(self.num_layers, batch, self.h_dim).type(dtype))

    def _forward(self, xy_in):
        """ Forward function of the Trajectory Encoder.

            Args:
                xy_in (tensor): Tensor of shape (in_len, batch_size, 2).

            Returns:
                tensor: Tensor of shape (self.num_layers, batch, self.h_dim).

            Note: The batchsize does not need to be static.
        """
        # Encode observed Trajectory
        batch = xy_in.size(1)
        xy_in_embedding = self.embedding(xy_in.contiguous().view(-1, 2))
        xy_in_embedding = xy_in_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(xy_in_embedding, state_tuple)
        hidden = state[0].view(-1, self.h_dim)
        return self.fc(hidden), self.fclogvar(hidden)

    def forward(self, xy_in):
        mu, logvar = self._forward(xy_in)
        std = logvar.mul(0.5).exp_()
        eps = get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar


class BicycleGenerator(nn.Module):

    def __init__(self, generator, embedding_dim=64, h_dim=64, z_dim=8, num_layers=1, dropout=0.0,
                 start_mode='clr', in_len=8, out_len=12, noise_type='gaussian', noise_mix_type='ped', **kwargs):
        super().__init__()
        noise_dim = (z_dim,) # required for sgan
        decoder_h_dim = h_dim + z_dim
        self.start_mode = start_mode

        self.generator = generator(in_len=in_len,
                                   out_len=out_len,
                                   embedding_dim=embedding_dim,
                                   encoder_h_dim=h_dim,
                                   z_dim=z_dim,
                                   num_layers=num_layers,
                                   noise_type=noise_type,
                                   noise_mix_type=noise_mix_type,
                                   dropout=dropout,
                                   noise_dim=noise_dim,
                                   decoder_h_dim=decoder_h_dim,
                                   **kwargs)

        self.encoder = BicycleEncoder(embedding_dim=embedding_dim,
                                      h_dim=h_dim,
                                      z_dim=z_dim,
                                      num_layers=num_layers,
                                      dropout=dropout)
        self.mode = start_mode
        self.z_dim = z_dim
        self.z_random = None
        self.mu = None
        self.logvar = None

    def forward(self, xy_in, dxdy_in, seq_start_end, xy_out=None):
        if self.mode == 'clr':
            return self.clr_forward(xy_in, dxdy_in, seq_start_end)
        elif self.mode == 'cvae':
            return self.cvae_forward(xy_in, dxdy_in, seq_start_end, xy_out)
        elif self.mode == 'eval':
            return self.generator(xy_in, dxdy_in, seq_start_end)
        else:
            raise AssertionError(f"self.type={self.mode} is invalid. Must be either 'clr', 'cvae' or 'eval'.")

    def clr_forward(self, xy_in, dxdy_in, seq_start_end):
        # With seq_start_end.size(0) instead of xy_in.size(1), along with global option in add_noise, noise would be
        # same for all agents within one sequence. Try if network doesnt train.
        self.z_random = get_z_random(xy_in.size(1), self.z_dim)
        dxdy_pred = self.generator(xy_in, dxdy_in, seq_start_end, user_noise=self.z_random)
        xy_pred = relative_to_abs(dxdy_pred, xy_in[-1])
        _, self.mu, self.logvar = self.encoder(xy_pred)  # mu~z_est
        return dxdy_pred

    def cvae_forward(self, xy_in, dxdy_in, seq_start_end, xy_out):
        z_encoded, self.mu, self.logvar = self.encoder(xy_out)
        dxdy_pred = self.generator(xy_in, dxdy_in, seq_start_end, user_noise=z_encoded)
        return dxdy_pred

    def eval(self):
        self.mode = 'eval'
        return self.train(False)

    def train(self, mode=True):
        if mode:
            self.mode=self.start_mode
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def clr(self):
        self.mode = 'clr'
        return self

    def cvae(self):
        self.mode = 'cvae'
        return self