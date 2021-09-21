from torch.nn import Embedding
import pytest

from .fixtures import *
from .modalities import InputModalityWithEmbedding
from .multi_modality_with_text_perceiver import MultiModalityWithTextPerceiver


def test_embedding_for_layer(text_inputs):
    text_modality = InputModalityWithEmbedding(
        name='text',
        input_channels=1,  # 1 channel for long ids representing tokens
        input_axis=1,  # number of axes, 2 for images
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
        embedding=Embedding(32000, text_embedding_dim)
    )
    assert text_inputs.size() == (3, 512,1)
    embedded = text_modality.embedding(text_inputs)
    assert embedded.size()==(3, 512,1, 256)
    assert text_modality.embedding_for_layer(embedded=embedded.squeeze(2), layer_index=0, depth=4).size() == (3, 512, 256//4)


def test_multimodality_forward_image_text(image_inputs,
                                          text_inputs,
                                          targets):
    image_modality = InputModalityWithEmbedding(
        name='image',
        input_channels=3,  # number of channels for each token of the input
        input_axis=2,  # number of axes, 2 for images
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
    )
    text_modality = InputModalityWithEmbedding(
        name='text',
        input_channels=1,  # 1 channel for long ids representing tokens
        input_axis=1,  # number of axes, 2 for images
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
        embedding=Embedding(32000, text_embedding_dim)
    )
    model = MultiModalityWithTextPerceiver(
        modalities=(image_modality, text_modality),
        depth=depth,  # depth of net
        num_latent_blocks_per_layer=2,
        num_latents=12,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=64,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=num_classes,  # output number of classes
        attn_dropout=0.,
        ff_dropout=0.,
        weight_tie_layers=True,
        # whether to weight tie layers (optional, as indicated in the diagram)
    )
    result = model({'image': image_inputs,
                    'text': text_inputs})
    assert result is not None
