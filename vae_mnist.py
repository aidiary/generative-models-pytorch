import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
        )

        self.mu_layer = nn.Linear(3136, 2)
        self.logvar_layer = nn.Linear(3136, 2)

    def forward(self, img):
        out = self.conv_layers(img)
        out = nn.Flatten()(out)
        mu = self.mu_layer(out)
        logvar = self.logvar_layer(out)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.dec_input_layer = nn.Linear(2, 3136)

    def forward(self, z):
        out = self.dec_input_layer(z)
        out = out.view(-1, 64, 7, 7)
        recon_imgs = self.deconv_layers(out)
        return recon_imgs


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, imgs):
        mu, logvar = self.encoder(imgs)
        z = self.reparameterize(mu, logvar)
        recon_imgs = self.decoder(z)
        return recon_imgs

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        imgs, _ = train_batch
        mu, logvar = self.encoder(imgs)
        z = self.reparameterize(mu, logvar)
        recon_imgs = self.decoder(z)

        recon_loss = F.mse_loss(recon_imgs, imgs)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        loss = recon_loss + kl_loss

        self.log('train/loss', loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train/kl_loss', kl_loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, _ = val_batch
        mu, logvar = self.encoder(imgs)
        z = self.reparameterize(mu, logvar)
        recon_imgs = self.decoder(z)

        recon_loss = F.mse_loss(recon_imgs, imgs)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        loss = recon_loss + kl_loss

        self.log('val/loss', loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val/kl_loss', kl_loss)

        return loss


if __name__ == '__main__':
    train_dataset = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    val_dataset = MNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              num_workers=8,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=32,
                            num_workers=8,
                            shuffle=False,
                            drop_last=True)

    model = VariationalAutoEncoder()

    tb_logger = TensorBoardLogger('lightning_logs', name='vae_mnist', default_hp_metric=False)
    trainer = pl.Trainer(gpus=[0], max_epochs=200, logger=tb_logger)
    trainer.fit(model, train_loader, val_loader)
