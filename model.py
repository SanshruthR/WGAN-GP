
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, d_dim=16):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, d_dim*32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(d_dim*32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d_dim*32, d_dim*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_dim*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d_dim*16, d_dim*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_dim*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d_dim*8, d_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_dim*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d_dim*4, d_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_dim*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d_dim*2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

class Critic(nn.Module):
    def __init__(self, d_dim=16):
        super(Critic, self).__init__()

        self.crit = nn.Sequential(
            nn.Conv2d(3, d_dim, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(d_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim, d_dim*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(d_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*2, d_dim*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(d_dim*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*4, d_dim*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(d_dim*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*8, d_dim*16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(d_dim*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d_dim*16, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
