import torch.nn as nn
import torch.nn.functional as F
import torch

import nnModels.nnModels_utils as nnModels_utils
from torch.autograd import Variable


class CVAE2D_ORIGINAL(nn.Module):
    """
    Convolutional 2D variational autoencoder, used to test the method.
    :attr: beta: regularisation term of the variational autoencoder. Increasing gamma gives more importance to the KL
    divergence term in the loss.
    :attr: gamma: Hyperparameter fixing the importance of the alignment loss in the total loss.
    :attr: latent_representation_size: size of the encoding given by the variational autoencoder
    :attr: name: name of the model
    """

    def __init__(self, latent_representation_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(CVAE2D_ORIGINAL, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 100
        self.lr = 1e-4  # For epochs between MCMC steps
        self.epoch = 0  # For tensorboard to keep track of total number of epochs
        self.name = 'CVAE_2D'

        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 16 x 32 x 32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 8 x 8
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc10 = nn.Linear(2048, latent_representation_size)
        self.fc11 = nn.Linear(2048, latent_representation_size)

        # self.fc2 = nn.Linear(8, 64)
        self.fc3 = nn.Linear(latent_representation_size, 512)
        self.upconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1)  # 32 x 16 x 16
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 16 x 32 x 32
        self.upconv3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1 x 64 x 64
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        # self.dropout = nn.Dropout(0.4)
        self.weight_init()

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        mu = torch.tanh(self.fc10(h3.flatten(start_dim=1)))
        # mu = (mu + torch.ones(mu.shape[1:])
        logVar = self.fc11(h3.flatten(start_dim=1))
        return mu, logVar

    def decoder(self, encoded):
        # h5 = F.relu(self.fc2(encoded))
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 8, 8, 8])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = torch.relu(self.upconv3(h8))
        return reconstructed

    def to(self, *args, **kwargs):
        new_self = super(CVAE2D_ORIGINAL, self).to(*args, **kwargs)
        device = next(self.parameters()).device
        self.device = device
        return new_self

    def reparametrize(self, mu, logVar):
        std = logVar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed, encoded

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block].modules():
                nnModels_utils.kaiming_init(m)

    def freeze_conv(self):
        """
        Freezes the convolutional layers.
        """
        # TODO: freeze bn as well ,
        nnModels_utils.freeze(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.upconv3])

    def unfreeze_conv(self):
        nnModels_utils.unfreeze(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.upconv3])


if __name__ == "__main__":
    model = CVAE2D_ORIGINAL(10)
