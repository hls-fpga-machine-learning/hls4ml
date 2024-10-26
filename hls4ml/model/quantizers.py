"""
This module contains definitions of hls4ml quantizer classes. These classes apply a quantization function on the
provided data. The quantization function may be defined locally or taken from a library in which case the classes
behave like simple wrappers.
"""

import numpy as np

from hls4ml.model.types import (
    ExponentPrecisionType,
    FixedPrecisionType,
    IntegerPrecisionType,
    RoundingMode,
    SaturationMode,
    XnorPrecisionType,
)


class Quantizer:
    """
    Base class for representing quantizers in hls4ml.

    Subclasses of ``Quantizer`` are expected to wrap the quantizers of upstream tools (e.g., QKeras).

    Args:
        bits (int): Total number of bits used by the quantizer.
        hls_type (NamedType): The hls4ml type used by the quantizer.
    """

    def __init__(self, bits, hls_type):
        self.bits = bits
        self.hls_type = hls_type

    def __call__(self, data):
        raise NotImplementedError


class BinaryQuantizer(Quantizer):
    """Quantizer that quantizes to 0 and 1 (``bits=1``) or -1 and 1 (``bits==2``).

    Args:
        bits (int, optional): Number of bits used by the quantizer. Defaults to 2.

    Raises:
        Exception: Raised if ``bits>2``
    """

    def __init__(self, bits=2):
        if bits == 1:
            hls_type = XnorPrecisionType()
        elif bits == 2:
            hls_type = IntegerPrecisionType(width=2)
        else:
            raise Exception(f'BinaryQuantizer suppots 1 or 2 bits, but called with bits={bits}')
        super().__init__(bits, hls_type)

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        quant_data = data
        if self.bits == 1:
            quant_data = np.where(data > 0, ones, zeros).astype('int')
        if self.bits == 2:
            quant_data = np.where(data > 0, ones, -ones)
        return quant_data


class TernaryQuantizer(Quantizer):
    """Quantizer that quantizes to -1, 0 and 1."""

    def __init__(self):
        super().__init__(2, IntegerPrecisionType(width=2))

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        return np.where(data > 0.5, ones, np.where(data <= -0.5, -ones, zeros))


class QKerasQuantizer(Quantizer):
    """Wrapper around QKeras quantizers.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config):
        from qkeras.quantizers import get_quantizer

        self.quantizer_fn = get_quantizer(config)
        self.alpha = config['config'].get('alpha', None)
        if config['class_name'] == 'quantized_bits':
            self.bits = config['config']['bits']
            self.hls_type = self._get_type(config)
        # ! includes stochastic_ternary
        elif 'ternary' in config['class_name']:
            self.bits = 2
            self.hls_type = IntegerPrecisionType(width=2, signed=True)
        # ! includes stochastic_binary
        elif 'binary' in config['class_name']:
            self.bits = 1
            self.hls_type = XnorPrecisionType()
        else:
            print('Unsupported quantizer: ' + config['class_name'])
            self.bits = 16
            self.hls_type = FixedPrecisionType(width=16, integer=6, signed=True)

    def __call__(self, data):
        data = np.array(data, dtype='float32')
        return self.quantizer_fn(data).numpy()
        # return self.quantizer_fn(data)

    def _get_type(self, quantizer_config):
        width = quantizer_config['config']['bits']
        integer = quantizer_config['config'].get('integer', 0)
        if quantizer_config['class_name'] == 'quantized_po2':
            return ExponentPrecisionType(width=width, signed=True)
        if width == integer:
            if width == 1:
                return XnorPrecisionType()
            else:
                return IntegerPrecisionType(width=width, signed=True)
        else:
            return FixedPrecisionType(width=width, integer=integer + 1, signed=True)


class QKerasBinaryQuantizer(Quantizer):
    """Wrapper around QKeras binary quantizer.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config, xnor=False):
        from qkeras.quantizers import get_quantizer

        self.bits = 1 if xnor else 2
        self.hls_type = XnorPrecisionType() if xnor else IntegerPrecisionType(width=2, signed=True)
        self.alpha = config['config']['alpha']
        # Use the QKeras quantizer to handle any stochastic / alpha stuff
        self.quantizer_fn = get_quantizer(config)
        # Then we use our BinaryQuantizer to convert to '0,1' format
        self.binary_quantizer = BinaryQuantizer(1) if xnor else BinaryQuantizer(2)

    def __call__(self, data):
        data = np.array(data, dtype='float32')
        y = self.quantizer_fn(data).numpy()
        return self.binary_quantizer(y)


class QKerasPO2Quantizer(Quantizer):
    """Wrapper around QKeras power-of-2 quantizers.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config):
        from qkeras.quantizers import get_quantizer

        self.bits = config['config']['bits']
        self.quantizer_fn = get_quantizer(config)
        self.hls_type = ExponentPrecisionType(width=self.bits, signed=True)

    def __call__(self, data):
        # Weights are quantized to nearest power of two
        data = np.array(data, dtype='float32')
        y = self.quantizer_fn(data)
        if hasattr(y, 'numpy'):
            y = y.numpy()
        return y


class QuantNodeQuantizer(Quantizer):
    """
    This implements a quantizer for a FixedPrecisionType with width==integer

    This is based on the sample implementation in finn-base
    """

    def __init__(self, precision):
        super().__init__(precision.width, precision)
        if not isinstance(precision, (FixedPrecisionType, IntegerPrecisionType)):
            raise TypeError('QuantNodeQuantizer is only defined for FixedPrecisionType and IntegerPrecisionType')

    def __call__(self, data):
        """Apply the quantization on the data"""

        scale = 2 ** (self.hls_type.width - self.hls_type.integer)

        data = data * scale  # (not using *= to avoid modifying data)
        # Clamping
        min_int_val = self._min_int(self.hls_type.signed, self.hls_type.saturation_mode, self.bits)
        max_int_val = self._max_int(self.hls_type.signed, self.bits)
        data = np.where(data > max_int_val, max_int_val, data)
        data = np.where(data < min_int_val, min_int_val, data)
        # Rounding
        rounding_fx = self._resolve_rounding_mode(self.hls_type.rounding_mode)
        return rounding_fx(data) / scale

    @staticmethod
    def _min_int(signed: bool, saturation_mode: str, bit_width: int) -> int:
        """Compute the minimum integer representable by a given number of bits.
        Args:
            signed (bool): Indicates whether the represented integer is signed or not.
            saturation_mode (bool): Indicates the saturation mode used (AP_SAT_SYM or AP_SAT)
            bit_width (int): Number of bits available for the representation.
        Returns:
            int: Maximum unsigned integer that can be represented according to
            the input arguments.
        Examples:
            >>> min_int(signed=True, saturation_mode='AP_SAT_SYM', bit_width=8)
            int(-127)
            >>> min_int(signed=False, saturation_mode='AP_SAT_SYM', bit_width=8)
            int(0)
            >>> min_int(signed=True, saturation_mode='AP_SAT', bit_width=8)
            int(-128)
            >>> min_int(signed=False, saturation_mode='AP_SAT_SYM', bit_width=8)
            int(0)
        """
        if saturation_mode not in (SaturationMode.SAT_SYM, SaturationMode.SAT, SaturationMode.WRAP):
            raise ValueError(
                f'Saturation mode {saturation_mode} not supported. Only AP_SAT_SYM, AP_SAT supported, WRAP partially'
            )
        if signed and saturation_mode == SaturationMode.SAT_SYM:
            value = -(2 ** (bit_width - 1)) + 1
        elif signed:
            value = -(2 ** (bit_width - 1))
        else:
            value = 0
        return value

    @staticmethod
    def _max_int(signed: bool, bit_width: int) -> int:
        """Compute the maximum integer representable by a given number of bits.
        (Note, narrow and unsigned is not supported by the implementation, so saturation mode is not used)
        Args:
            signed (bool): Indicates whether the represented integer is signed or not.
            bit_width (int): Number of bits available for the representation.
        Returns:
            Tensor: Maximum integer that can be represented according to
            the input arguments.
        Examples:
            >>> max_int(signed=True, bit_width=8)
            int(127)
            >>> max_int(signed=False, bit_width=8)
            int(255)
        """
        if not signed:
            value = (2**bit_width) - 1
        else:
            value = (2 ** (bit_width - 1)) - 1
        return value

    @staticmethod
    def _resolve_rounding_mode(mode):
        """Resolve the rounding mode  of Quant and Trunc ops
        to the corresponding numpy functions."""
        if mode == RoundingMode.RND_CONV:
            return np.round
        # elif mode_string == 'CEIL':   # not supported
        #     return np.ceil
        elif mode == RoundingMode.TRN:
            return np.floor
        else:
            raise ValueError(f'Rounding mode {mode} not supported.')
