import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, img):
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        logit = self.conv5(out)
        return logit


class WGAN_GP(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        # lightningで複数のoptimizerがあるときはmanual更新にする
        self.automatic_optimization = False

    def forward(self, img):
        pass

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        return opt_g, opt_d

    def d_loss_fn(self, r_logit, f_logit):
        return -r_logit.mean() + f_logit.mean()

    def g_loss_fn(self, f_logit):
        return -f_logit.mean()

    def gradient_penalty(self, real_img, fake_img):
        sample_img = self.sample(real_img, fake_img)
        sample_img.requires_grad = True
        pred = self.discriminator(sample_img)
        grad = torch.autograd.grad(pred,
                                   sample_img,
                                   grad_outputs=torch.ones_like(pred),
                                   create_graph=True)[0]

        grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)
        gradient_penalty = ((grad_norm - 1)**2).mean()

        return gradient_penalty

    def sample(self, real_img, fake_img):
        shape = [real_img.shape[0], 1, 1, 1]
        alpha = torch.rand(shape, device=real_img.device)
        sample = alpha * fake_img + (1 - alpha) * real_img
        return sample

    def training_step(self, train_batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        real_img, _ = train_batch
        batch_size = real_img.shape[0]

        # train discriminator
        z = torch.randn(batch_size, 128, 1, 1, device=self.device)
        fake_img = self.generator(z).detach()

        real_d_logit = self.discriminator(real_img)
        fake_d_logit = self.discriminator(fake_img)

        d_loss = self.d_loss_fn(real_d_logit, fake_d_logit)
        gp = self.gradient_penalty(real_img, fake_img)

        d_loss = d_loss + 10.0 * gp
        self.log('train/d_loss', d_loss)
        self.log('train/gp', gp)

        opt_d.zero_grad()
        self.manual_backward(d_loss, retain_graph=True)
        opt_d.step()

        # 5 iterationごとにgeneratorを訓練
        if self.trainer.global_step % 5 == 0:
            z = torch.randn(batch_size, 128, 1, 1, device=self.device)
            fake_img = self.generator(z)

            fake_d_logit = self.discriminator(fake_img)
            g_loss = self.g_loss_fn(fake_d_logit)
            self.log('train/g_loss', g_loss)

            opt_g.zero_grad()
            self.manual_backward(g_loss, retain_graph=True)
            opt_g.step()


if __name__ == '__main__':
    # data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CelebA(root='data', split='train', transform=transform, download=False)
    val_dataset = CelebA(root='data', split='test', transform=transform, download=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              num_workers=8,
                              shuffle=True,
                              drop_last=True)

    # model
    model = WGAN_GP()

    # training
    tb_logger = TensorBoardLogger('lightning_logs', name='wgan_gp_celeba', default_hp_metric=False)
    trainer = pl.Trainer(gpus=[0], max_epochs=100, logger=tb_logger)
    trainer.fit(model, train_loader)
