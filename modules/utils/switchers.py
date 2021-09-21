from modules.networks import AHnet2d, AHnet3d, MedNet, VNet, DenseNet3D, SkipDenseNet3D, ResNet3D_VAE, HighResNet3D, HyperDensenet, DenseVoxelNet, DenseNet3D
from modules.networks.unet.unet import UNet3DClassification
from modules.networks.EfficientNet3D.model import EfficientNet3D

switcher_network = {
            'MedNet10': MedNet.resnet10,
            'MedNet18': MedNet.resnet18,
            'MedNet34': MedNet.resnet34,
            'MedNet50': MedNet.resnet50,
            'MedNet101': MedNet.resnet101,
            'MedNet152': MedNet.resnet152,
            'MedNet200': MedNet.resnet200,
            'AHNet3D': AHnet3d.AHNet,
            'AHNet2D': AHnet2d.MCFCN,
            'UNet3D': UNet3DClassification,
            'VNet': VNet.VNet,
            'DenseNet3DSinglePath': DenseNet3D.SinglePathDenseNet,
            'SkipDenseNet3D': SkipDenseNet3D.SkipDenseNet3D,
            'ResNet3dVAE': ResNet3D_VAE.ResNet3dVAE,
            'HighResNet3D': HighResNet3D.HighResNet3D,
            'EfficientNet3D-B0': EfficientNet3D,
            'EfficientNet3D-B1': EfficientNet3D,
            'EfficientNet3D-B2': EfficientNet3D,
            'EfficientNet3D-B3': EfficientNet3D,
            'EfficientNet3D-B4': EfficientNet3D,
            'EfficientNet3D-B5': EfficientNet3D,
            'EfficientNet3D-B6': EfficientNet3D,
            'EfficientNet3D-B7': EfficientNet3D,
            'HyperDenseNet': HyperDensenet.HyperDenseNet,
            'DenseVoxelNet': DenseVoxelNet.DenseVoxelNet,
            'SinglePathDenseNet': DenseNet3D.SinglePathDenseNet
}

