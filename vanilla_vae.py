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

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.fc_mu = nn.Linear(512 * 2 * 2, 128)
        self.fc_var = nn.Linear(512 * 2 * 2, 128)

    def forward(self, img):
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = nn.Flatten()(out)

        mu = self.fc_mu(out)
        logvar = self.fc_var(out)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_input = nn.Linear(128, 2048)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.decoder_input(z)

        out = out.view(-1, 512, 2, 2)

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
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        img, labels = train_batch

        mu, logvar = self.encoder(img)
        z = self.reparameterize(mu, logvar)
        recon_img = self.decoder(z)

        recon_loss = F.mse_loss(recon_img, img)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        loss = recon_loss + kld_loss

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

    def sample(self):
        pass


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
    trainer = pl.Trainer(gpus=[1], max_epochs=50)
    trainer.fit(model, train_loader, val_loader)
