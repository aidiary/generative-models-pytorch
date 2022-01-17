import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1,
                      stride=1,
                      bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(self.num_residual_layers)
        ])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens // 2,
                                kernel_size=4,
                                stride=2,
                                padding=1)
        self.conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                out_channels=num_hiddens,
                                kernel_size=4,
                                stride=2,
                                padding=1)
        self.conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.conv_3(x)
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)

        self.conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                               out_channels=num_hiddens // 2,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)

        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                               out_channels=3,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.residual_stack(x)
        x = self.conv_trans_1(x)
        x = F.relu(x)
        return self.conv_trans_2(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],
                                self.num_embeddings,
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss
        # detach = sg (stop gradient)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # こうすることでqantizedの部分がskipされてinputsにgradが流れる
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return vq_loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VQVAE(pl.LightningModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                 embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1,
                                     stride=1)
        self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers,
                               num_residual_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        vq_loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)
        return vq_loss, x_recon, perplexity

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, amsgrad=False)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        data, _ = train_batch

        vq_loss, data_recon, perplexity = self.forward(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss

        self.log('train/recon_error', recon_error)
        self.log('train/vq_loss', vq_loss)
        self.log('train/perplexity', perplexity)

        return loss


if __name__ == '__main__':
    training_data = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))]),
    )

    validation_data = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))]),
    )

    data_variance = np.var(training_data.data / 255.0)

    training_loader = DataLoader(
        training_data,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    model = VQVAE(num_hiddens=128,
                  num_residual_hiddens=32,
                  num_residual_layers=2,
                  num_embeddings=512,
                  embedding_dim=64,
                  commitment_cost=0.25)
    tb_logger = TensorBoardLogger('./lightning_logs', name='vqvae_cifar10', default_hp_metric=False)
    trainer = pl.Trainer(gpus=[0], logger=tb_logger, max_steps=15000)
    trainer.fit(model, training_loader)
