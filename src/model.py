from monai.networks.nets import UNet

def get_model(config):
    model = UNet(
        spatial_dims=3,
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        channels=(16, 32, 64, 128, 256), # Standard U-Net depth
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
    return model