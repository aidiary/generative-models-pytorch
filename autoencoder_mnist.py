import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import MNIST


class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
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
            nn.Flatten(),
            nn.Linear(3136, 2),
        )

        self.decoder = nn.Sequential(
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

    def forward(self, imgs):
        z = self.encoder(imgs)
        z = self.dec_input_layer(z)
        z = z.view(-1, 64, 7, 7)
        recon_imgs = self.decoder(z)
        return recon_imgs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        imgs, labels = train_batch
        recon_imgs = self.forward(imgs)

        loss = F.mse_loss(recon_imgs, imgs)

        self.log('train/loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, labels = val_batch
        recon_imgs = self.forward(imgs)

        loss = F.mse_loss(recon_imgs, imgs)

        self.log('valid/loss', loss)

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

    model = AutoEncoder()

    tb_logger = TensorBoardLogger('lightning_logs', name='autoencoder', default_hp_metric=False)
    trainer = pl.Trainer(gpus=[0], max_epochs=200, logger=tb_logger)
    trainer.fit(model, train_loader, val_loader)
