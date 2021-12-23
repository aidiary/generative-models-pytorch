import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 2048),
            nn.BatchNorm1d(2048, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        self.mu_layer = nn.Linear(2048, 128)
        self.var_layer = nn.Linear(2048, 128)

    def forward(self, imgs):
        out = self.conv_layers(imgs)
        out = nn.Flatten()(out)
        out = self.fc(out)
        mu = self.mu_layer(out)
        logvar = self.var_layer(out)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 256 * 8 * 8, bias=False),
            nn.BatchNorm1d(256 * 8 * 8, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, 256, 8, 8)
        recon_imgs = self.deconv_layers(out)
        return recon_imgs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, imgs):
        out = self.conv_layers(imgs)
        out = nn.Flatten()(out)
        bottleneck = out
        out = self.fc(out)
        return out, bottleneck


class VAE_GAN(pl.LightningModule):
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

        # lightningで複数のoptimizerがあるときはmanual更新にする
        self.automatic_optimization = False

    def forward(self, imgs):
        mu, logvar = self.encoder(imgs)
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        recon_imgs = self.decoder(z)
        return mu, logvar, recon_imgs

    def configure_optimizers(self):
        lr = 3e-4
        optim_encoder = optim.Adam(self.encoder.parameters(), lr=lr)
        optim_decoder = optim.Adam(self.decoder.parameters(), lr=lr)
        optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr)
        return optim_encoder, optim_decoder, optim_discriminator

    def training_step(self, train_batch, batch_idx):
        opt_enc, opt_dec, opt_dis = self.optimizers()

        imgs, _ = train_batch
        mu, logvar, recon_imgs = self.forward(imgs)

        # gan_lossにはサンプリングした画像からも含めるとよい
        # muとlogvarではなく　N(0, I) からサンプリングして画像を復元
        z_p = torch.randn_like(mu)
        x_p_tilda = self.decoder(z_p)

        # discriminatorの訓練 (gan_loss)
        # real dataは1として判定したい
        out, _ = self.discriminator(imgs)
        loss_d_real = self.bce_loss(out, torch.ones_like(out))

        # fake dataは0と判定したい
        out, _ = self.discriminator(recon_imgs)
        loss_d_fake = self.bce_loss(out, torch.zeros_like(out))

        # samplingした画像は0と判定したい
        out, _ = self.discriminator(x_p_tilda)
        loss_d_sample = self.bce_loss(out, torch.zeros_like(out))

        gan_loss = loss_d_real + loss_d_fake + loss_d_sample
        opt_dis.zero_grad()
        self.manual_backward(gan_loss, retain_graph=True)
        opt_dis.step()

        # decoder (= generator) の訓練 (gamma * recon_loss - gan_loss)
        # gan_lossはあとでマイナス記号をつけるので最大化を目指すことになる
        # TODO: realがzeros_like、fake/sampleがones_likeで最小化のほうがわかりやすい
        out, _ = self.discriminator(imgs)
        loss_d_real = self.bce_loss(out, torch.ones_like(out))
        out, _ = self.discriminator(recon_imgs)
        loss_d_fake = self.bce_loss(out, torch.zeros_like(out))
        out, _ = self.discriminator(x_p_tilda)
        loss_d_sample = self.bce_loss(out, torch.zeros_like(out))
        gan_loss2 = loss_d_real + loss_d_fake + loss_d_sample

        # discriminatorのボトルネック特徴量の距離でreconstruction loss
        _, x_l = self.discriminator(imgs)
        _, x_l_tilda = self.discriminator(recon_imgs)
        recon_loss = self.mse_loss(x_l, x_l_tilda)
        gamma = 15.0
        dec_loss = gamma * recon_loss - gan_loss2
        opt_dec.zero_grad()
        self.manual_backward(dec_loss, retain_graph=True)
        opt_dec.step()

        # encoderの訓練 (kl_loss + recon_loss)
        # ganのlossは使わない
        # descriminatorのボトルネック特徴量の距離でreconstruction loss
        mu, logvar, recon_imgs = self.forward(imgs)
        _, x_l = self.discriminator(imgs)
        _, x_l_tilda = self.discriminator(recon_imgs)
        recon_loss = self.mse_loss(x_l, x_l_tilda)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / torch.numel(mu.data)
        enc_loss = 5.0 * recon_loss + kl_loss
        opt_enc.zero_grad()
        self.manual_backward(enc_loss, retain_graph=True)
        opt_enc.step()

        enc_loss = enc_loss.item()
        dec_loss = dec_loss.item()
        gan_loss = gan_loss.item()

        self.log('train/enc_loss', enc_loss)
        self.log('train/dec_loss', dec_loss)
        self.log('train/gan_loss', gan_loss)

        return {'enc_loss': enc_loss, 'dec_loss': dec_loss, 'gan_loss': gan_loss}

    def validation_step(self, val_batch, batch_idx):
        imgs, _ = val_batch
        mu, logvar, recon_imgs = self.forward(imgs)

        # muとlogvarではなく　N(0, I) からサンプリングして画像を復元
        z_p = torch.randn_like(mu)
        x_p_tilda = self.decoder(z_p)

        # discriminator
        # real dataは1として判定したい
        out, _ = self.discriminator(imgs)
        loss_d_real = self.bce_loss(out, torch.ones_like(out))

        # fake dataは0と判定したい
        out, _ = self.discriminator(recon_imgs)
        loss_d_fake = self.bce_loss(out, torch.zeros_like(out))

        # samplingした画像は0と判定したい
        out, _ = self.discriminator(x_p_tilda)
        loss_d_sample = self.bce_loss(out, torch.zeros_like(out))

        gan_loss = loss_d_real + loss_d_fake + loss_d_sample

        # decoder (= generator)
        # gan_lossはあとでマイナス記号をつけるので最大化を目指すことになる
        # TODO: realがzeros_like、fake/sampleがones_likeで最小化のほうがわかりやすい
        out, _ = self.discriminator(imgs)
        loss_d_real = self.bce_loss(out, torch.ones_like(out))
        out, _ = self.discriminator(recon_imgs)
        loss_d_fake = self.bce_loss(out, torch.zeros_like(out))
        out, _ = self.discriminator(x_p_tilda)
        loss_d_sample = self.bce_loss(out, torch.zeros_like(out))
        gan_loss2 = loss_d_real + loss_d_fake + loss_d_sample

        # discriminatorのボトルネック特徴量の距離でreconstruction loss
        _, x_l = self.discriminator(imgs)
        _, x_l_tilda = self.discriminator(recon_imgs)
        recon_loss = self.mse_loss(x_l, x_l_tilda)
        gamma = 15.0
        dec_loss = gamma * recon_loss - gan_loss2

        # encoderの訓練
        # descriminatorのボトルネック特徴量の距離でreconstruction loss
        mu, logvar, recon_imgs = self.forward(imgs)
        _, x_l = self.discriminator(imgs)
        recon_loss = self.mse_loss(x_l, x_l_tilda)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / torch.numel(mu.data)
        enc_loss = 5.0 * recon_loss + kl_loss

        enc_loss = enc_loss.item()
        dec_loss = dec_loss.item()
        gan_loss = gan_loss.item()

        self.log('val/enc_loss', enc_loss)
        self.log('val/dec_loss', dec_loss)
        self.log('val/gan_loss', gan_loss)

        return {'enc_loss': enc_loss, 'dec_loss': dec_loss, 'gan_loss': gan_loss}

    def reconstruct(self, img):
        mu, _ = self.encoder(img)
        recon_imgs = self.decoder(mu)
        return recon_imgs

    def sample(self, num_samples=64):
        z = torch.randn(num_samples, 128)
        samples = self.decoder(z)
        return samples


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_dataset = CelebA(root='data', split='train', transform=transform, download=False)
    val_dataset = CelebA(root='data', split='test', transform=transform, download=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              num_workers=8,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=64,
                            num_workers=8,
                            shuffle=False,
                            drop_last=True)

    model = VAE_GAN()

    # training
    tb_logger = TensorBoardLogger('lightning_logs', name='vae_gan', default_hp_metric=False)
    trainer = pl.Trainer(gpus=[0], max_epochs=200, logger=tb_logger)
    trainer.fit(model, train_loader, val_loader)
