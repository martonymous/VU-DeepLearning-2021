import torch.nn as nn

nc = 3
lat_vector_size = 100

featmap_size_generator = 64
featmap_size_discriminator = 16

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        def generator_block(in_filters, out_filters, kernel_size=(4,4), stride=(2,2), padding=(1,1), bn=True):
            block = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding, bias=False), nn.ReLU(True)]
            if bn:
                pass# block.insert(-2, nn.BatchNorm2d(out_filters))
            return block

        self.main = nn.Sequential(
            # input size = z
            *generator_block(lat_vector_size, featmap_size_generator * 4, stride=1, padding=0),
            *generator_block(featmap_size_generator * 4, featmap_size_generator * 2),
            *generator_block(featmap_size_generator * 2, featmap_size_generator),
            nn.ConvTranspose2d(featmap_size_generator, nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        def discriminator_block(in_filters, out_filters, kernel_size=(4,4), stride=(2,2), padding=(1,1), bn=True, lr=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=False)]
            if bn:
                pass# block.append(nn.BatchNorm2d(out_filters))
            if lr:
                block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.main = nn.Sequential(

            *discriminator_block(nc, featmap_size_discriminator, bn=False),                         # output = (fsd) x 32 x 32
            *discriminator_block(featmap_size_discriminator, featmap_size_discriminator * 2),       # output = (2fsd) x 16 x 16
            *discriminator_block(featmap_size_discriminator * 2, featmap_size_discriminator * 4),   # output = (4fsd) x 8 x 8
            *discriminator_block(featmap_size_discriminator * 4, 1, 4, 1, 0, bn=False, lr=False),   # output = 1
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
