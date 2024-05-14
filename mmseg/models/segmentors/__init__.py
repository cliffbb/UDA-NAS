# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add 'EncoderDecoderSearch'

from .base import BaseSegmentor
from .encoder_decoder_search import EncoderDecoderSearch
from .encoder_decoder import EncoderDecoder

__all__ = ['BaseSegmentor', 'EncoderDecoderSearch', 'EncoderDecoder']
