from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image
from utils import *

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)

UNIQUE_RUN_ID = str(uuid.uuid4())
make_directory_for_run(UNIQUE_RUN_ID)

random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 100
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 32
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 16
# Number of training epochs
num_epochs = 500
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
                                ])
# dataset load
# full_dataset = dset.MNIST(root='./data', train=True, download=True, transform=transform)
# test = dset.MNIST(root='./data', train=False, download=True, transform=transform)
svhn = dset.SVHN(root='./data', download=True, transform=transform)

"""get train_test_split"""
sub_size = int(0.8 * len(svhn))
subm_size = len(svhn) - sub_size
train, test = torch.utils.data.random_split(svhn, [sub_size, subm_size])

# Create the dataloader
dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
# Create the dataloader
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# initialize generator and discriminator
netG, netD = initialize_models(device, ngpu)

# Initialize BCELoss function
criterion = nn.BCELoss()

"""Create batch of latent vectors that we will use to visualize the progression of the generator"""
fixed_noise = generate_noise(nz, 64, device=device)
"""Establish convention for real and fake labels during training"""
real_label = 1.
fake_label = 0.
"""Setup Adam optimizers for both G and D"""
optimizerG, optimizerD = initialize_optimizers(netG, netD, lr, (beta1, 0.999), lr, (beta1, 0.999))

# Training Loop
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print(netG, '\n\n', netD)
print("Starting Training Loop...")

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        """
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        """

        """Train with all-real batch"""
        netD.zero_grad()

        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        """Train with all-fake batch"""
        # Generate batch of latent vectors
        noise = generate_noise(nz, b_size)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        """
        # (2) Update G network: maximize log(D(G(z)))
        """
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        """Output training stats"""
        if i % 5 == 0:
            save_training_image(fixed_noise, generator=netG, UNIQUE_RUN_ID=UNIQUE_RUN_ID, epoch=epoch, batch=i)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


        save_models(generator=netG, discriminator=netD, epoch=epoch, UNIQUE_RUN_ID=UNIQUE_RUN_ID)

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

