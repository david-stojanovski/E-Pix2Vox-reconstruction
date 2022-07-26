import torch


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU()
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )
        self.layer11 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer12 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_volumes):
        volumes_128_l = coarse_volumes.view((-1, 1, 128, 128, 128))

        volumes_64_l = self.layer1(volumes_128_l)
        volumes_32_l = self.layer2(volumes_64_l)
        volumes_16_l = self.layer3(volumes_32_l)
        volumes_8_l = self.layer4(volumes_16_l)
        volumes_4_l = self.layer5(volumes_8_l)

        flatten_features = self.layer6(volumes_4_l.view(-1, 8192))
        flatten_features = self.layer7(flatten_features)
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)

        volumes_8_r = volumes_8_l + self.layer8(volumes_4_r)
        volumes_16_r = volumes_16_l + self.layer9(volumes_8_r)
        volumes_32_r = volumes_32_l + self.layer10(volumes_16_r)
        volumes_64_r = volumes_64_l + self.layer11(volumes_32_r)
        volumes_128_r = (volumes_128_l + self.layer12(volumes_64_r)) * 0.5

        return volumes_128_r.view((-1, 128, 128, 128))
