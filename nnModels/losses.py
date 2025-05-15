import torch
import torch.nn.functional as F


def longitudinal_loss(encoded, predicted):
    return torch.sum((encoded - predicted) ** 2) / predicted.shape[0]


def spatial_auto_encoder_loss(mu, logVar, reconstructed, input_):
    kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
    # recon_error = torch.nn.MSELoss(reduction='mean')(reconstructed, input_)
    recon_error = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
    return recon_error, kl_divergence


def spatial_2D_auto_encoder_loss(mu, logVar, reconstructed, input_):
    print(reconstructed, input_)
    kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
    # Binary Cross Entropy
    recon_error = F.binary_cross_entropy(reconstructed, input_, size_average=False) / input_.shape[0]

    return recon_error, kl_divergence


def pixel_reconstruction_error(mu, logVar, reconstructed, input):
    """
    Assume that input and reconstructed are 1 image (not a batch of images).
    """
    recon_error = torch.abs(input - reconstructed)
    return recon_error, None


def loss_bvae(recon_x, x, mu, logvar, beta):
    batch_size = x.size(0)
    bce = F.binary_cross_entropy(recon_x, x, size_average=False).div(batch_size)
    # bce = F.mse_loss(recon_x, x, size_average=False).div(batch_size)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(1).mean(0, True)

    return bce + beta * kld


def loss_bvae2(mu, logvar, recon_x, x):
    batch_size = x.size(0)
    bce = F.binary_cross_entropy(recon_x, x, size_average=False).div(batch_size)
    # bce = F.mse_loss(recon_x, x, size_average=False).div(batch_size)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(1).mean(0, True)

    return bce, kld


if __name__ == "__main__":
    mu = torch.Tensor([1, 2])
    logvar = torch.Tensor([[3, 4], [5, 6]])
    image = torch.Tensor([[0, 1], [1, 1], [1, 1]])
    reconstructed = torch.Tensor([[0.5, 0.9], [0.10, 0.16], [0.18, 0.13]])
    beta = 5
    bce, kl = loss_bvae2(mu, logvar, reconstructed, image)
    print(bce + beta * kl, loss_bvae(reconstructed, image, mu, logvar, beta))
    print(loss_bvae2(mu, logvar, reconstructed, image), spatial_2D_auto_encoder_loss(mu, logvar, reconstructed, image))
