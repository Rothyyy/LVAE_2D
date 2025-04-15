import torch

from longitudinalModel.train_with_crops_as_input import train_with_crops_as_input


class LongitudinalVAE():
    def __init__(self, nnmodel=None, nnModel_optimizer=None, longitudinal_estimator=None,
                 longitudinal_estimator_settings=None,
                 longitudinal_estimator_optimization_settings=None, lr=1e-5, beta=5, gamma=100,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.nnmodel = nnmodel
        self.nnModel_optimizer = nnModel_optimizer
        self.longitudinal_estimator = longitudinal_estimator
        self.longitudinal_estimator_settings = longitudinal_estimator_settings
        self.longitudinal_estimator_optimization_settings = longitudinal_estimator_optimization_settings
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.device = device

    def train(self, data_loader, nn_saving_path='model.pth',
              longitudinal_saving_path='longitudinal_estimator_params.json', loss_graph_saving_path=None):
        train_with_crops_as_input(self.nnmodel, data_loader,
                                  latent_representation_size=self.nnmodel.latent_representation_size,
                                  longitudinal_estimator=self.longitudinal_estimator,
                                  longitudinal_estimator_settings=None, encodings_csv_path=None, nb_epochs=100, lr=0.01,
                                  device=self.device, nn_saving_path=nn_saving_path,
                                  longitudinal_saving_path=longitudinal_saving_path,
                                  loss_graph_saving_path=loss_graph_saving_path)

    def test(self):
        return

    def display_results(self):
        return

    def decode(self, z):
        self.nnmodel.decode(z)

    def encode(self, x):
        self.nnmodel.encode(x)

    def encode_at_age(self):
        return
