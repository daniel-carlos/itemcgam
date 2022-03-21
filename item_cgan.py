import signal
import sys
import itertools

from models import Generator, Discriminator
from loaders import ItemDataset

import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torch

os.makedirs("images", exist_ok=True)
os.makedirs("results/images", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=30,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400,
                    help="interval between image sampling")
parser.add_argument("--json_data", type=str,
                    default="./dataset/items/items.json", help="path to json data")
parser.add_argument("--imgs_path", type=str,
                    default="./dataset/items/images", help="path to images")
parser.add_argument("--resume", type=bool, default=False,
                    help="resume last session")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
if opt.resume:
    generator = torch.load("results/models/item_cgan_g.pt")
    discriminator = torch.load("results/models/item_cgan_d.pt")
    generator.eval()
    discriminator.eval()
else:
    generator = Generator(opt.latent_dim, 5, img_shape)
    discriminator = Discriminator(5, img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
transform = transforms.Grayscale() if opt.channels == 1 else None
dataloader = DataLoader(
    ItemDataset(opt.json_data, opt.imgs_path, opt.img_size, transform),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(batches_done):
    # Sample noise
    all_possibs = [
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [
            0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 0, 1], [
                0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [
                0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1], [
                0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [
            1, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1], [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [
                1, 0, 1, 1, 0], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1], [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [
                1, 1, 0, 1, 0], [1, 1, 0, 1, 0], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [
                1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]
    ]
    z = FloatTensor(np.random.normal(0, 1, (len(all_possibs), opt.latent_dim)))

    np_arr = np.array(all_possibs).reshape((len(all_possibs), 5))
    # Gerar todas as possibilidades
    labelsx = torch.from_numpy(np_arr)

    gen_imgs = generator(z, labelsx, img_shape)
    save_image(gen_imgs.data, "results/images/%d.png" %
               batches_done, nrow=8, normalize=True)


def save_checkpoint():
    torch.save(generator, "results/models/item_cgan_g.pt")
    torch.save(discriminator, "results/models/item_cgan_d.pt")

def load_checkpoint():
    pass #TODO:load checkpoint

def signal_handler(signal, frame):
    save_checkpoint
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ----------
#  Training
# ----------

starting_epoch = 0

for epoch in range(starting_epoch, opt.n_epochs):
    for i, sample in enumerate(dataloader):
        imgs = sample['image']
        batch_size = imgs.shape[0]

        masks = sample['mask']

        # Adversarial ground truths
        valid = FloatTensor(batch_size, 1).fill_(1.0)
        fake = FloatTensor(batch_size, 1).fill_(0.0)

        # Configure input
        real_imgs = imgs.type(FloatTensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))
        gen_mask = torch.randint(0, 2, (batch_size, 5))

        # Generate a batch of images
        gen_imgs = generator(z, gen_mask, img_shape)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_mask)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, masks)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_mask)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(batches_done=batches_done)

signal_handler(0, 0)
