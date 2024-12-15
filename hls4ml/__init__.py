from hls4ml import converters, report, utils  # noqa: F401, E402

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


def reseed(newseed):
    print(f'\npytest-randomly: reseed with {newseed}')
    try:
        import tensorflow

        tensorflow.random.set_seed(newseed)
    except ImportError:
        print('\nTensorFlow seed not set')
    try:
        import torch

        torch.manual_seed(newseed)
    except ImportError:
        print('\nPyTorch seed not set')
