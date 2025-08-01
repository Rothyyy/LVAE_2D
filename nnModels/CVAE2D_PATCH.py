import torch.nn as nn
import torch.nn.functional as F
import torch

import nnModels.nnModels_utils as nnModels_utils
from torch.autograd import Variable


class CVAE2D_PATCH(nn.Module):
    """
    Convolutional 2D variational autoencoder, used to test the method.
    :attr: beta: regularisation term of the variational autoencoder. Increasing gamma gives more importance to the KL
    divergence term in the loss.
    :attr: gamma: Hyperparameter fixing the importance of the alignment loss in the total loss.
    :attr: latent_representation_size: size of the encoding given by the variational autoencoder
    :attr: name: name of the model
    """

    def __init__(self, latent_representation_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(CVAE2D_PATCH, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 100
        self.lr = 1e-4  # For epochs between MCMC steps
        self.epoch = 0  # For tensorboard to keep track of total number of epochs
        self.name = 'CVAE_2D_PATCH'

        # Encoder
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, stride=1) 
        self.conv2 = nn.Conv2d(3, 4, kernel_size=3, stride=1) 
        self.conv3 = nn.Conv2d(4, 12, kernel_size=3, stride=3)
        self.conv4 = nn.Conv2d(12, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(12)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc5 = nn.Linear(16, latent_representation_size)   
        self.fc6 = nn.Linear(16, latent_representation_size)   
        # self.fc11 = nn.Linear(288, latent_representation_size)

        # Decoder
        self.fc_decode = nn.Linear(latent_representation_size, 16) 
        self.deconv1 = nn.ConvTranspose2d(16, 12, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(12, 4, kernel_size=3, stride=3) 
        self.deconv3 = nn.ConvTranspose2d(4, 3, kernel_size=3, stride=1) 
        self.deconv4 = nn.ConvTranspose2d(3, 1, kernel_size=5, stride=1) 
        self.bn5 = nn.BatchNorm2d(12)
        self.bn6 = nn.BatchNorm2d(4)
        self.bn7 = nn.BatchNorm2d(3)
        self.bn8 = nn.BatchNorm2d(1)

        self.weight_init()

    def encoder(self, patch):
        # print("Start encoder")
        h1 = F.gelu(self.bn1(self.conv1(patch)))
        # print("h1 shape =", h1.shape)
        h2 = F.gelu(self.bn2(self.conv2(h1)))
        # print("h2 shape =", h2.shape)
        h3 = F.gelu(self.bn3(self.conv3(h2)))
        # print("h3 shape =", h3.shape)
        h4 = F.gelu(self.bn4(self.conv4(h3)))
        # print("h4 shape =", h4.shape)
        mu = self.fc5(h4.flatten(start_dim=1))
        logvar = self.fc6(h4.flatten(start_dim=1))
        # print("mu shape =", mu.shape)
        # print("logvar shape =", logvar.shape)
        return mu, logvar

    def decoder(self, encoded):
        # print("Start decoder")
        # print("Shape encoding =", encoded.shape)
        h5 = self.fc_decode(encoded).view(-1, 16, 1, 1)
        # print("Shape h5 =", h5.shape)
        h6 = F.gelu(self.bn5(self.deconv1(h5))) 
        # print("Shape h6 =", h6.shape)
        h7 = F.gelu(self.bn6(self.deconv2(h6))) 
        # print("Shape h7 =", h7.shape)
        h8 = F.gelu(self.bn7(self.deconv3(h7))) 
        # print("Shape h8 =", h8.shape)
        # h9 = F.gelu(self.bn8(self.deconv4(h8)))       # VERSION output in [0,1]
        h9 = F.tanh(self.bn8(self.deconv4(h8)))     # VERSION output in [-1,1]
        # print("Shape h9 =", h9.shape)

        return h9

    def to(self, *args, **kwargs):
        new_self = super(CVAE2D_PATCH, self).to(*args, **kwargs)
        device = next(self.parameters()).device
        self.device = device
        return new_self

    def reparametrize(self, mu, logVar):
        std = logVar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(self, image):
        # print("Start forward")
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
    
    def freeze_all(self):
        """
        Freezes the convolutional layers.
        """
        # TODO: freeze bn as well ,
        nnModels_utils.freeze(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.upconv3, self.bn1, self.bn2, self.bn3, self.bn4, self.bn5])

    def unfreeze_conv(self):
        nnModels_utils.unfreeze(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.upconv3])



class CVAE2D_PATCH_new(nn.Module):
    """
    Convolutional 2D variational autoencoder, used to test the method.
    :attr: beta: regularisation term of the variational autoencoder. Increasing gamma gives more importance to the KL
    divergence term in the loss.
    :attr: gamma: Hyperparameter fixing the importance of the alignment loss in the total loss.
    :attr: latent_representation_size: size of the encoding given by the variational autoencoder
    :attr: name: name of the model
    """

    def __init__(self, latent_representation_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(CVAE2D_PATCH_new, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 100
        self.lr = 1e-4  # For epochs between MCMC steps
        self.epoch = 0  # For tensorboard to keep track of total number of epochs
        self.name = 'CVAE_2D_PATCH'

        # Encoder
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, stride=1)       # [3, 11, 11]
        self.conv2 = nn.Conv2d(3, 4, kernel_size=3, stride=1)       # [4, 9, 9]
        self.conv3 = nn.Conv2d(4, 12, kernel_size=3, stride=3)      # [12, 3, 3]
        self.conv4_mu = nn.Conv2d(12, 16, kernel_size=3, stride=1)  # [16, 1 ,1]
        self.conv4_var = nn.Conv2d(12, 16, kernel_size=3, stride=1) # [16, 1 ,1]

        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(12)
        self.bn4_mu = nn.BatchNorm2d(16)
        self.bn4_var = nn.BatchNorm2d(16)

        # self.fc5 = nn.Linear(16, latent_representation_size)   
        # self.fc6 = nn.Linear(16, latent_representation_size)   
        # self.fc11 = nn.Linear(288, latent_representation_size)

        # Decoder
        # self.fc_decode = nn.Linear(latent_representation_size, 16) 
        self.deconv1 = nn.ConvTranspose2d(16, 12, kernel_size=3, stride=1)  # [12, 3, 3]
        self.deconv2 = nn.ConvTranspose2d(12, 4, kernel_size=3, stride=3)   # [4, 9, 9]
        self.deconv3 = nn.ConvTranspose2d(4, 3, kernel_size=3, stride=1)    # [3, 11 ,11]
        self.deconv4 = nn.ConvTranspose2d(3, 1, kernel_size=5, stride=1)    # [1, 15, 15]
        self.bn5 = nn.BatchNorm2d(12)
        self.bn6 = nn.BatchNorm2d(4)
        self.bn7 = nn.BatchNorm2d(3)
        self.bn8 = nn.BatchNorm2d(1)

        self.weight_init()

    def encoder(self, patch):
        # print("Start encoder")
        h1 = F.gelu(self.bn1(self.conv1(patch)))
        # print("h1 shape =", h1.shape)
        h2 = F.gelu(self.bn2(self.conv2(h1)))
        # print("h2 shape =", h2.shape)
        h3 = F.gelu(self.bn3(self.conv3(h2)))
        # print("h3 shape =", h3.shape)
        mu = F.gelu(self.bn4_mu(self.conv4_mu(h3)))
        logvar = F.gelu(self.bn4_var(self.conv4_var(h3)))
        # print("mu shape =", mu.shape)
        # print("logvar shape =", logvar.shape)
        return mu, logvar

    def decoder(self, encoded):
        # print("Start decoder")
        # print("Shape encoding =", encoded.shape)
        # h5 = self.fc_decode(encoded).view(-1, 16, 1, 1)
        # print("Shape h5 =", h5.shape)
        h6 = F.gelu(self.bn5(self.deconv1(encoded))) 
        # print("Shape h6 =", h6.shape)
        h7 = F.gelu(self.bn6(self.deconv2(h6))) 
        # print("Shape h7 =", h7.shape)
        h8 = F.gelu(self.bn7(self.deconv3(h7))) 
        # print("Shape h8 =", h8.shape)
        h9 = F.sigmoid(self.bn8(self.deconv4(h8)))       # VERSION output in [0,1]
        # h9 = F.tanh(self.bn8(self.deconv4(h8)))     # VERSION output in [-1,1]
        # print("Shape h9 =", h9.shape)

        return h9

    def to(self, *args, **kwargs):
        new_self = super(CVAE2D_PATCH_new, self).to(*args, **kwargs)
        device = next(self.parameters()).device
        self.device = device
        return new_self

    def reparametrize(self, mu, logVar):
        std = logVar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def forward(self, image):
        # print("Start forward")
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
    
    def freeze_all(self):
        """
        Freezes the convolutional layers.
        """
        # TODO: freeze bn as well ,
        nnModels_utils.freeze(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.upconv3, self.bn1, self.bn2, self.bn3, self.bn4, self.bn5])

    def unfreeze_conv(self):
        nnModels_utils.unfreeze(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.upconv3])


if __name__ == "__main__":
    model = CVAE2D_PATCH(4)
