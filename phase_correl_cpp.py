"""
Phase Correlation implementation translated from C++ to Python.
Provides exact functional equivalence to the original phase_correl.cpp.
"""

import math
import sys
from typing import List, Tuple
import copy


class GrayscaleImage:
    """Grayscale image class with basic drawing operations."""

    def __init__(self, width: int = 0, height: int = 0, fill: int = 0, *, _source: 'GrayscaleImage' = None):
        """
        Initialize a GrayscaleImage.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            fill: Fill value for all pixels (0-255)
            _source: Internal parameter for copy construction
        """
        if _source is not None:
            # Copy constructor behavior
            self._data = list(_source._data)
            self._width = _source._width
            self._height = _source._height
        else:
            # Normal constructor: allocate and fill with specified value
            self._data = [fill] * (width * height)
            self._width = width
            self._height = height

    @classmethod
    def from_copy(cls, source: 'GrayscaleImage') -> 'GrayscaleImage':
        """Copy constructor equivalent."""
        return cls(_source=source)

    def copy_from(self, source: 'GrayscaleImage') -> 'GrayscaleImage':
        """Assignment operator equivalent."""
        self._data = list(source._data)
        self._width = source._width
        self._height = source._height
        return self

    def draw_rectangle(self, x: int, y: int, width: int, height: int, fill: int) -> None:
        """
        Draw a filled rectangle on the image.

        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Rectangle width
            height: Rectangle height
            fill: Fill value (0-255)
        """
        for j in range(y, y + height):
            for i in range(x, x + width):
                if (i < self._width) and (j < self._height):
                    self._data[i + j * self._width] = fill

    def get_width(self) -> int:
        """Get image width."""
        return self._width

    def get_height(self) -> int:
        """Get image height."""
        return self._height

    def get_data(self) -> List[int]:
        """Get image data as a list of pixel values."""
        return self._data

    def set(self, width: int, height: int, data: List[int]) -> None:
        """
        Set image data.

        Args:
            width: New image width
            height: New image height
            data: New pixel data
        """
        self._data = list(data)
        self._width = width
        self._height = height


class PhaseCorrelation:
    """
    Phase correlation class for computing image shift.
    All methods are static - class cannot be instantiated.
    """

    def __init__(self):
        """Deleted constructor - class cannot be instantiated."""
        raise RuntimeError("PhaseCorrelation cannot be instantiated")

    @staticmethod
    def compute_shift(image1: GrayscaleImage, image2: GrayscaleImage) -> Tuple[int, int]:
        """
        Compute the shift between two images using phase correlation.

        Args:
            image1: First grayscale image
            image2: Second grayscale image

        Returns:
            Tuple of (deltax, deltay) representing the shift

        Raises:
            RuntimeError: If image sizes do not match or are not powers of 2
        """
        if ((image1.get_width() != image2.get_width()) or
            (image1.get_height() != image2.get_height()) or
            not image1.get_width() or
            (image1.get_width() & (image1.get_width() - 1)) or
            not image1.get_height() or
            (image1.get_height() & (image1.get_height() - 1))):
            raise RuntimeError("Image sizes do not match")

        if (not image1.get_width() or
            (image1.get_width() & (image1.get_width() - 1)) or
            not image1.get_height() or
            (image1.get_height() & (image1.get_height() - 1))):
            raise RuntimeError("Wrong image size")

        width = image1.get_width()
        height = image1.get_height()

        data1: List[complex] = [complex(0, 0)] * (width * height)
        data2: List[complex] = [complex(0, 0)] * (width * height)

        # Convert image pixels to complex number format, use only real part
        for i in range(width * height):
            data1[i] = complex(float(image1.get_data()[i]), 0.0)
            data2[i] = complex(float(image2.get_data()[i]), 0.0)

        # Perform 2D FFT on each image
        PhaseCorrelation._fft_2d(data1, width)
        PhaseCorrelation._fft_2d(data2, width)

        # Compute normalized cross power spectrum
        for i in range(width * height):
            data1[i] = PhaseCorrelation._comp_norm_cross_correlation(data1[i], data2[i])

        # Perform inverse 2D FFT on obtained matrix
        PhaseCorrelation._fft_2d(data1, width, True)

        # Search for peak
        offset = 0
        max_val = 0.0
        deltax = 0
        deltay = 0
        for j in range(height):
            for i in range(width):
                d = math.sqrt(pow(data1[offset].real, 2) + pow(data1[offset].imag, 2))
                if d > max_val:
                    max_val = d
                    deltax = i
                    deltay = j
                offset += 1

        if deltax > (width >> 1):
            deltax -= width
        if deltay > (height >> 1):
            deltay -= height

        return (deltax, deltay)

    @staticmethod
    def _comp_norm_cross_correlation(input1: complex, input2: complex) -> complex:
        """
        Compute normalized cross correlation for a single complex pair.

        Args:
            input1: First complex number
            input2: Second complex number

        Returns:
            Normalized cross correlation result as complex number
        """
        # std::arg(input1 * std::conj(input2)) - argument (phase) of product with conjugate
        diff = cmath_arg(input1 * input2.conjugate())

        return complex(math.cos(diff), math.sin(diff))

    @staticmethod
    def _radix2_fft(input_data: List[complex], input_offset: int, output: List[complex], output_offset: int, stride: int) -> None:
        """
        Radix-2 FFT butterfly operation.

        Args:
            input_data: Input complex array
            input_offset: Offset into input array
            output: Output complex array
            output_offset: Offset into output array
            stride: Stride between input elements
        """
        output[output_offset] = input_data[input_offset] + input_data[input_offset + stride]
        output[output_offset + 1] = input_data[input_offset] - input_data[input_offset + stride]

    @staticmethod
    def _sum2_fft(input_data: List[complex], input_offset: int, output: List[complex], output_offset: int, size: int, inverse: bool) -> None:
        """
        Sum two FFT results with twiddle factors.

        Args:
            input_data: Input complex array
            input_offset: Offset into input array
            output: Output complex array
            output_offset: Offset into output array
            size: Size of the FFT segment
            inverse: True for inverse FFT
        """
        dfi = (2.0 if inverse else -2.0) * math.pi / float(size << 1)
        kfi = 0.0

        for k in range(size):
            cosfi = math.cos(kfi)
            sinfi = math.sin(kfi)
            temp = [input_data[input_offset + k], input_data[input_offset + k + size]]

            output[output_offset + k] = complex(
                temp[0].real + cosfi * temp[1].real - sinfi * temp[1].imag,
                temp[0].imag + sinfi * temp[1].real + cosfi * temp[1].imag
            )
            output[output_offset + k + size] = complex(
                temp[0].real - cosfi * temp[1].real + sinfi * temp[1].imag,
                temp[0].imag - sinfi * temp[1].real - cosfi * temp[1].imag
            )
            kfi += dfi

    @staticmethod
    def _gen_input_order(size: int) -> List[int]:
        """
        Generate bit-reversed input order for FFT.

        Args:
            size: Size of the FFT

        Returns:
            List of indices in bit-reversed order
        """
        stride = 1
        # SIZE_MAX equivalent - use a large sentinel value
        SIZE_MAX = (2 ** 63) - 1
        offset = [SIZE_MAX] * size
        offset[0] = 0

        step = size >> 1
        while step > 0:
            base = 0
            for i in range(0, size, step):
                if offset[i] != SIZE_MAX:
                    base = offset[i]
                else:
                    offset[i] = base + stride
            stride <<= 1
            step >>= 1

        return offset

    @staticmethod
    def _dit2_fft(input_data: List[complex], inverse: bool = False) -> List[complex]:
        """
        Decimation-in-time radix-2 FFT.

        Args:
            input_data: Input complex array
            inverse: True for inverse FFT

        Returns:
            Output complex array
        """
        stride = 1
        size = len(input_data)
        offset = PhaseCorrelation._gen_input_order(size >> 1)
        output = [complex(0, 0)] * size

        while size > 2:
            stride <<= 1
            size >>= 1

        for i in range(stride):
            PhaseCorrelation._radix2_fft(input_data, offset[i], output, i * 2, stride)

        while stride > 1:
            stride >>= 1
            size <<= 1
            for i in range(stride):
                PhaseCorrelation._sum2_fft(output, i * size, output, i * size, size >> 1, inverse)

        if inverse:
            for i in range(size):
                output[i] = output[i] / float(size)

        return output

    @staticmethod
    def _fft_2d(data: List[complex], width: int, inverse: bool = False) -> None:
        """
        2D FFT in-place.

        Args:
            data: Complex data array (modified in place)
            width: Width of the 2D array
            inverse: True for inverse FFT
        """
        height = len(data) // width

        def horizontal_fft():
            """Perform FFT along columns (vertical direction in memory layout)."""
            input_vec = [complex(0, 0)] * height
            for i in range(width):
                offset = i
                for j in range(height):
                    input_vec[j] = data[offset]
                    offset += width
                output_vec = PhaseCorrelation._dit2_fft(input_vec, inverse)
                offset = i
                for j in range(height):
                    data[offset] = output_vec[j]
                    offset += width

        def vertical_fft():
            """Perform FFT along rows (horizontal direction in memory layout)."""
            input_vec = [complex(0, 0)] * width
            for j in range(height):
                offset = j * width
                for i in range(width):
                    input_vec[i] = data[offset]
                    offset += 1
                output_vec = PhaseCorrelation._dit2_fft(input_vec, inverse)
                offset = j * width
                for i in range(width):
                    data[offset] = output_vec[i]
                    offset += 1

        if not inverse:
            vertical_fft()
            horizontal_fft()
        else:
            horizontal_fft()
            vertical_fft()


def cmath_arg(z: complex) -> float:
    """
    Compute the argument (phase angle) of a complex number.
    Equivalent to std::arg() in C++.

    Args:
        z: Complex number

    Returns:
        Phase angle in radians
    """
    return math.atan2(z.imag, z.real)


def main() -> int:
    """Main function demonstrating phase correlation."""
    try:
        # Generate pair of images
        image1 = GrayscaleImage(256, 128, 0x00)
        image2 = GrayscaleImage(256, 128, 0xff)
        image1.draw_rectangle(16, 32, 60, 60, 0x80)
        image2.draw_rectangle(8, 40, 60, 60, 0x10)

        deltax, deltay = PhaseCorrelation.compute_shift(image1, image2)
    except Exception:
        print("Operation failed", file=sys.stderr)
        return 1

    print(f"Calculated shift: [{deltax}, {deltay}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
