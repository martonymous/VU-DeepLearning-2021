import torch.nn as nn
import torch, os,math, uuid
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from modules import *


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def initialize_models(device, ngpu):
    """ Initialize Generator and Discriminator models """
    generator = Generator(ngpu)
    discriminator = Discriminator(ngpu)
    # Perform proper weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    # Move models to specific device
    generator.to(device)
    discriminator.to(device)
    # Return models
    return generator, discriminator

def initialize_optimizers(generator, discriminator,g_lr, g_betas, d_lr, d_betas):
    """ Initialize optimizers for Generator and Discriminator. """
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=g_lr, betas=g_betas)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=d_lr, betas=d_betas)
    return generator_optimizer, discriminator_optimizer

def generate_noise(noise_dimension, number_of_images=1, device=None):
    """ Generate noise for number_of_images images, and with noise_dimension """
    return torch.randn(number_of_images, noise_dimension, 1, 1, device=device)

def save_training_image(noise, generator, UNIQUE_RUN_ID, epoch=0, batch=0, show=True):
    """ Generate subplots with generated examples. """
    images = generator(noise)

    if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/images'):
        os.makedirs(f'./runs/{UNIQUE_RUN_ID}/images')

    name = f'./runs/{UNIQUE_RUN_ID}/images/epoch{epoch}_batch{batch}.jpg'
    save_image(images, name, nrow=10)

def make_directory_for_run(UNIQUE_RUN_ID):
    """ Make a directory for this training run. """
    print(f'Preparing training run {UNIQUE_RUN_ID}')
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
    os.mkdir(f'./runs/{UNIQUE_RUN_ID}')

def save_models(generator, discriminator, epoch, UNIQUE_RUN_ID):
    """ Save models at specific point in time. """
    torch.save(generator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/generator_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/discriminator_{epoch}.pth')

def output_shape(input_tensor: tuple, kernel_size: tuple, stride: tuple, padding: tuple, batch_size=1, output_channel=1):
    return batch_size, \
           output_channel, \
           math.floor((input_tensor[-2] - kernel_size[-2] + (2*padding[-2]) + stride[-2]) / stride[-2]), \
           math.floor((input_tensor[-1] - kernel_size[-1] + (2*padding[-1]) + stride[-1]) / stride[-1])

def interpolate(src, dest, steps=9):
    '''Linearly interpolate between two vectors.'''
    step = (dest - src) / steps
    return torch.cat([src + (step * i) for i in range(steps + 1)], dim=0)

def sample_latent(gan, start, stop, points=10):
    samples = interpolate(start, stop, points - 1)
    outputs = gan(samples)
    return outputs