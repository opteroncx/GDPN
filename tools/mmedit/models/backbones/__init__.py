# Copyright (c) OpenMMLab. All rights reserved.
from .encoder_decoders import (VGG16, ContextualAttentionNeck, DeepFillDecoder,
                               DeepFillEncoder, DeepFillEncoderDecoder,
                               DepthwiseIndexBlock, FBADecoder,
                               FBAResnetDilated, GLDecoder, GLDilationNeck,
                               GLEncoder, GLEncoderDecoder, HolisticIndexBlock,
                               IndexedUpsample, IndexNetDecoder,
                               IndexNetEncoder, PConvDecoder, PConvEncoder,
                               PConvEncoderDecoder, PlainDecoder,
                               ResGCADecoder, ResGCAEncoder, ResNetDec,
                               ResNetEnc, ResShortcutDec, ResShortcutEnc,
                               SimpleEncoderDecoder)
from .generation_backbones import ResnetGenerator, UnetGenerator
from .sr_backbones import (GLEANStyleGANv2,GLEANStyleGANv2Edge,
                           MSRResNet, RRDBNet)

__all__ = [
    'MSRResNet', 'VGG16', 'PlainDecoder', 'SimpleEncoderDecoder',
    'GLEncoderDecoder', 'GLEncoder', 'GLDecoder', 'GLDilationNeck',
    'PConvEncoderDecoder', 'PConvEncoder', 'PConvDecoder', 'ResNetEnc',
    'ResNetDec', 'ResShortcutEnc', 'ResShortcutDec', 'RRDBNet',
    'DeepFillEncoder', 'HolisticIndexBlock', 'DepthwiseIndexBlock',
    'ContextualAttentionNeck', 'DeepFillDecoder',
    'DeepFillEncoderDecoder', 'IndexedUpsample', 'IndexNetEncoder',
    'IndexNetDecoder', 'ResGCAEncoder', 'ResGCADecoder', 
    'UnetGenerator', 'ResnetGenerator', 'FBAResnetDilated', 'FBADecoder',
    'GLEANStyleGANv2','GLEANStyleGANv2Edge'
]
