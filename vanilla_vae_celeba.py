import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.mu_layer = nn.Linear(4096, 200)
        self.logvar_layer = nn.Linear(4096, 200)

    def forward(self, imgs):
        out = self.conv_layers(imgs)

        out = nn.Flatten()(out)

        mu = self.mu_layer(out)
        logvar = self.logvar_layer(out)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_input = nn.Linear(200, 4096)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.decoder_input(z)

        out = out.view(-1, 64, 8, 8)

        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        recon_img = self.deconv5(out)

        return recon_img


class VanillaVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img):
        mu, logvar = self.encoder(img)
        return mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img, labels = train_batch

        mu, logvar = self.encoder(img)
        z = self.reparameterize(mu, logvar)
        recon_img = self.decoder(z)

        recon_loss = F.mse_loss(recon_img, img)
        kld_weight = 1e-3
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + kld_weight * kld_loss

        self.log('train/loss', loss)
        self.log('train/recon_loss', recon_loss)
        self.log('train/kld_loss', kld_loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        img, labels = val_batch

        mu, logvar = self.encoder(img)
        z = self.reparameterize(mu, logvar)
        recon_img = self.decoder(z)

        recon_loss = F.mse_loss(recon_img, img)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        loss = recon_loss + kld_loss

        self.log('val/loss', loss)
        self.log('val/recon_loss', recon_loss)
        self.log('val/kld_loss', kld_loss)

        return loss

    def reconstruct(self, img):
        mu, _ = self.encoder(img)
        recon_img = self.decoder(mu)
        return recon_img

    def sample(self, num_samples=144):
        z = torch.randn(num_samples, 128)
        samples = self.decoder(z)
        return samples


if __name__ == '__main__':
    # data
    SetRange = transforms.Lambda(lambda x: 2 * x - 1)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor(),
        SetRange,
    ])

    train_dataset = CelebA(root='data', split='train', transform=transform, download=False)
    val_dataset = CelebA(root='data', split='test', transform=transform, download=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=144,
                              num_workers=8,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=144,
                            num_workers=8,
                            shuffle=False,
                            drop_last=True)

    # model
    model = VanillaVAE()

    # training
    trainer = pl.Trainer(gpus=[0], max_epochs=50)
    trainer.fit(model, train_loader, val_loader)
