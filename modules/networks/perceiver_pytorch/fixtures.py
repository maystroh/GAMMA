import torch
from pytest import fixture

batch_size = 3
num_classes = 32
depth = 2
text_embedding_dim=256

@fixture()
def targets():
    # batch of 3, 32 frames, 3 channels each frame 260 x 260
    targets = torch.randint(high=num_classes, size=(batch_size, 1), requires_grad=False).view(-1)
    return targets


@fixture()
def image_inputs():
    return torch.rand(size=(3, 260, 260, 3), requires_grad=True)


@fixture()
def video_inputs():
    # batch of 3, 32 frames, 3 channels each frame 260 x 260
    return torch.rand(size=(3, 32, 260, 260, 3), requires_grad=True)

@fixture()
def audio_inputs():
    # one second of audio sampled at 44100 (one channel/mono)
    return torch.rand(size=(3, 44100,1), requires_grad=True)


@fixture()
def text_inputs():
    # text token ids of length 512 (1 channel). 32000 tokens.
    return torch.randint(high=32000, size=(3, 512, 1)).long()

