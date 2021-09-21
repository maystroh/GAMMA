from fixtures import *
from .hierarchical_multi_modality_perceiver import HierarchicalMultiModalityPerceiver, \
    HierarchicalConfigurator, HierarchicalMultiModalityPerceiverNoPooling
from .modalities import modality_encoding
from .multi_modality_perceiver import MultiModalityPerceiver, InputModality, \
    MultiModalityPerceiverNoPooling


def test_modality_encoding():
    x = modality_encoding(batch_size=3, axes=(32, 12), modality_index=0, num_modalities=2)
    assert x.size() == (3, 32, 12, 2)


def test_hierarchical_multimodality_forward_image_video(image_inputs, video_inputs, audio_inputs,
                                           targets):
    video_modality = InputModality(
        name='video',
        input_channels=3,  # number of channels for each token of the input
        input_axis=3,  # number of axes, 3 for video)
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
    )
    image_modality = InputModality(
        name='image',
        input_channels=3,  # number of channels for each token of the input
        input_axis=2,  # number of axes, 2 for images
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
    )
    audio_modality = InputModality(
        name='audio',
        input_channels=1,  # number of channels for mono audio
        input_axis=1,  # number of axes, 2 for images
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )
    model = HierarchicalMultiModalityPerceiver(
        modalities=(video_modality, image_modality, audio_modality),
        depth=depth,  # depth of net

        # number of latents, or induced set points, or centroids. different papers giving it different names

        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=num_classes,  # output number of classes
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False,
        num_latent_blocks_per_layer=1,
        configurator=HierarchicalConfigurator(num_latents_begin=12,latent_dim_begin=24,
                                              depth=depth)
        # whether to weight tie layers (optional, as indicated in the diagram)
    )
    print(model)
    result = model({'image': image_inputs,
                    'video': video_inputs,
                    'audio': audio_inputs})
    assert result is not None


def test_multimodality_forward_image_video_no_pooling(image_inputs, video_inputs, audio_inputs,
                                                      targets):
    video_modality = InputModality(
        name='video',
        input_channels=3,  # number of channels for each token of the input
        input_axis=3,  # number of axes, 3 for video)
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
    )
    image_modality = InputModality(
        name='image',
        input_channels=3,  # number of channels for each token of the input
        input_axis=2,  # number of axes, 2 for images
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
    )
    audio_modality = InputModality(
        name='audio',
        input_channels=1,  # number of channels for mono audio
        input_axis=1,  # number of axes, 2 for images
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )
    num_latents = 12
    latent_dim = 17

    configurator = HierarchicalConfigurator(num_latents_begin=num_latents, latent_dim_begin=latent_dim, depth=depth)
    assert configurator.get_num_latents(depth - 1) == 6
    assert configurator.get_latent_dim(depth - 1) == 34
    model = HierarchicalMultiModalityPerceiverNoPooling(
        modalities=(video_modality, image_modality, audio_modality),
        depth=depth,  # depth of net
        num_latents=num_latents,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=latent_dim,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=False,
        configurator=configurator
        # whether to weight tie layers (optional, as indicated in the diagram)
    )

    result = model({'image': image_inputs,
                    'video': video_inputs,
                    'audio': audio_inputs})
    assert result is not None

    assert result.size() == (image_inputs.size()[0], configurator.get_num_latents(depth-1), configurator.get_latent_dim(depth-1))
