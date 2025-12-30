#!/usr/bin/env python3
"""
Phase Correlation implementation - Direct translation from C

This module provides a direct line-by-line translation of the C implementation
of phase correlation for image shift detection using FFT.

Algorithms implemented:
- Radix-2 FFT (Decimation in Time)
- Sum2FFT for combining FFT stages
- 2D FFT for image processing
- Normalized cross-correlation for phase detection
"""

import math
import sys
from typing import List, Tuple, Optional


def radix2fft(input_arr: List[float], output: List[float], input_offset: int, output_offset: int, stride: int) -> None:
    """
    Radix-2 FFT butterfly operation.

    Performs the basic 2-point FFT butterfly computation.
    Complex numbers are stored as interleaved real/imaginary pairs.

    Args:
        input_arr: Input array with complex values (interleaved real/imag)
        output: Output array with complex values (interleaved real/imag)
        input_offset: Starting offset in input array
        output_offset: Starting offset in output array
        stride: Stride for accessing second input element
    """
    output[output_offset + 0] = input_arr[input_offset + 0] + input_arr[input_offset + (stride << 1)]
    output[output_offset + 1] = input_arr[input_offset + 1] + input_arr[input_offset + (stride << 1) + 1]

    output[output_offset + 2] = input_arr[input_offset + 0] - input_arr[input_offset + (stride << 1)]
    output[output_offset + 3] = input_arr[input_offset + 1] - input_arr[input_offset + (stride << 1) + 1]


def sum2fft(input_arr: List[float], output: List[float], offset: int, size: int, inverse: int) -> None:
    """
    Combine two FFT halves using twiddle factors.

    Combines the results of smaller FFTs using complex exponential
    (twiddle) factors to produce the next stage of the FFT.

    Args:
        input_arr: Input array with complex values (interleaved real/imag)
        output: Output array with complex values (interleaved real/imag)
        offset: Starting offset in arrays
        size: Half the size of the current FFT stage
        inverse: 1 for inverse FFT, 0 for forward FFT
    """
    temp = [0.0, 0.0, 0.0, 0.0]
    dfi = (2.0 if inverse else -2.0) * math.pi / float(size << 1)
    kfi = 0.0

    for k in range(size):
        cosfi = math.cos(kfi)
        sinfi = math.sin(kfi)
        temp[0] = input_arr[offset + (k << 1)]
        temp[1] = input_arr[offset + (k << 1) + 1]
        temp[2] = input_arr[offset + ((k + size) << 1)]
        temp[3] = input_arr[offset + ((k + size) << 1) + 1]

        output[offset + (k << 1)] = temp[0] + cosfi * temp[2] - sinfi * temp[3]
        output[offset + (k << 1) + 1] = temp[1] + sinfi * temp[2] + cosfi * temp[3]
        output[offset + ((k + size) << 1)] = temp[0] - cosfi * temp[2] + sinfi * temp[3]
        output[offset + ((k + size) << 1) + 1] = temp[1] - sinfi * temp[2] - cosfi * temp[3]
        kfi += dfi


def gen2fftorder(size: int) -> List[int]:
    """
    Generate bit-reversal order for FFT.

    Creates an array of offsets that define the order in which
    input samples should be processed for the decimation-in-time FFT.

    Args:
        size: Number of complex samples

    Returns:
        List of offsets for reordering
    """
    # Initialize with sentinel value (equivalent to memset 0xff / SIZE_MAX)
    SIZE_MAX = -1  # Use -1 as sentinel (equivalent to SIZE_MAX check)
    offset = [SIZE_MAX] * size
    stride = 1

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


def dit2fft(input_arr: List[float], output: List[float], size: int, inverse: int) -> None:
    """
    Decimation-in-Time FFT.

    Computes the FFT using the Cooley-Tukey decimation-in-time algorithm.
    Input and output are arrays of interleaved complex values (real, imag, real, imag, ...).

    Args:
        input_arr: Input array with complex values (interleaved real/imag)
        output: Output array with complex values (interleaved real/imag)
        size: Number of complex samples
        inverse: 1 for inverse FFT, 0 for forward FFT
    """
    offset = gen2fftorder(size >> 1)
    stride = 1
    working_size = size

    while working_size > 2:
        stride <<= 1
        working_size >>= 1

    for i in range(stride):
        radix2fft(input_arr, output, offset[i] << 1, (i * 2) << 1, stride)

    # offset is automatically garbage collected (equivalent to free)

    while stride > 1:
        stride >>= 1
        working_size <<= 1
        for i in range(stride):
            sum2fft(output, output, (i * working_size) << 1, working_size >> 1, inverse)

    if inverse:
        for i in range(working_size):
            output[i << 1] /= float(working_size)
            output[(i << 1) + 1] /= float(working_size)


def fft2d(data: List[float], width: int, height: int, inverse: int) -> int:
    """
    2D FFT for image processing.

    Performs a 2D FFT by applying 1D FFTs along rows and columns.
    For forward FFT: rows first, then columns.
    For inverse FFT: columns first, then rows.

    Args:
        data: Array with complex values (interleaved real/imag), modified in-place
        width: Image width (must be power of 2)
        height: Image height (must be power of 2)
        inverse: 1 for inverse FFT, 0 for forward FFT

    Returns:
        0 on success, -1 on error
    """
    # Allocate working buffers
    try:
        fft_input = [0.0] * (width << 1)
        fft_output = [0.0] * (width << 1)
    except MemoryError:
        return -1

    if inverse:
        # For inverse: horizontal (columns) first
        pass
    else:
        # For forward: vertical (rows) first
        # vertical_fft:
        for j in range(height):
            offset = j * width
            for i in range(width):
                fft_input[i << 1] = data[offset << 1]
                fft_input[(i << 1) + 1] = data[(offset << 1) + 1]
                offset += 1
            dit2fft(fft_input, fft_output, width, inverse)
            offset = j * width
            for i in range(width):
                data[offset << 1] = fft_output[i << 1]
                data[(offset << 1) + 1] = fft_output[(i << 1) + 1]
                offset += 1

    # horizontal_fft:
    # Reallocate for height if different from width
    if height != width:
        fft_input = [0.0] * (height << 1)
        fft_output = [0.0] * (height << 1)

    for i in range(width):
        offset = i
        for j in range(height):
            fft_input[j << 1] = data[offset << 1]
            fft_input[(j << 1) + 1] = data[(offset << 1) + 1]
            offset += width
        dit2fft(fft_input, fft_output, height, inverse)
        offset = i
        for j in range(height):
            data[offset << 1] = fft_output[j << 1]
            data[(offset << 1) + 1] = fft_output[(j << 1) + 1]
            offset += width

    if inverse:
        # vertical_fft for inverse:
        # Reallocate for width if different from height
        if height != width:
            fft_input = [0.0] * (width << 1)
            fft_output = [0.0] * (width << 1)

        for j in range(height):
            offset = j * width
            for i in range(width):
                fft_input[i << 1] = data[offset << 1]
                fft_input[(i << 1) + 1] = data[(offset << 1) + 1]
                offset += 1
            dit2fft(fft_input, fft_output, width, inverse)
            offset = j * width
            for i in range(width):
                data[offset << 1] = fft_output[i << 1]
                data[(offset << 1) + 1] = fft_output[(i << 1) + 1]
                offset += 1

    # done:
    # fft_input and fft_output are automatically garbage collected (equivalent to free)

    return 0


def comp_norm_cross_correlation(f: List[float], g: List[float], r: List[float],
                                 f_offset: int, g_offset: int, r_offset: int) -> None:
    """
    Compute normalized cross-correlation for complex values.

    Computes the phase difference between two complex numbers and
    returns a unit complex number with that phase.

    Args:
        f: First complex array (interleaved real/imag)
        g: Second complex array (interleaved real/imag)
        r: Result complex array (interleaved real/imag)
        f_offset: Offset in f array
        g_offset: Offset in g array
        r_offset: Offset in r array
    """
    diff = math.atan2(f[f_offset + 1] * g[g_offset + 0] - f[f_offset + 0] * g[g_offset + 1],
                      f[f_offset + 0] * g[g_offset + 0] + f[f_offset + 1] * g[g_offset + 1])

    r[r_offset + 0] = math.cos(diff)
    r[r_offset + 1] = math.sin(diff)


def compute_shift(image1: List[int], image2: List[int], width: int, height: int) -> Tuple[int, int, int]:
    """
    Compute the shift between two images using phase correlation.

    Uses the phase correlation method to determine the translational
    offset between two images. Both width and height must be powers of 2.

    Args:
        image1: First image as list of pixel values (0-255)
        image2: Second image as list of pixel values (0-255)
        width: Image width (must be power of 2)
        height: Image height (must be power of 2)

    Returns:
        Tuple of (return_code, deltax, deltay)
        return_code: 0 on success, -1 on error
        deltax: Horizontal shift
        deltay: Vertical shift
    """
    # Check power of 2 for width and height
    if not width or (width & (width - 1)) or not height or (height & (height - 1)):
        return (-1, 0, 0)

    ret = 0
    deltax = 0
    deltay = 0

    # Allocate FFT buffers
    try:
        fft_input1 = [0.0] * ((width * height) << 1)
        fft_input2 = [0.0] * ((width * height) << 1)
        fft_output = [0.0] * ((width * height) << 1)
    except MemoryError:
        return (-1, 0, 0)

    # Convert image pixels to complex number format, use only real part
    for i in range(width * height):
        fft_input1[i << 1] = float(image1[i])
        fft_input2[i << 1] = float(image2[i])

        fft_input1[(i << 1) + 1] = 0.0
        fft_input2[(i << 1) + 1] = 0.0

    # Perform 2D FFT on each image
    ret = fft2d(fft_input1, width, height, 0)
    if ret:
        return (ret, deltax, deltay)
    ret = fft2d(fft_input2, width, height, 0)
    if ret:
        return (ret, deltax, deltay)

    # Compute normalized cross power spectrum
    for i in range(width * height):
        comp_norm_cross_correlation(fft_input1, fft_input2, fft_output, i << 1, i << 1, i << 1)

    # Perform inverse 2D FFT on obtained matrix
    ret = fft2d(fft_output, width, height, 1)
    if ret:
        return (ret, deltax, deltay)

    # Search for peak
    offset = 0
    max_val = 0.0
    deltax = 0
    deltay = 0
    for j in range(height):
        for i in range(width):
            d = math.sqrt(pow(fft_output[offset << 1], 2) + pow(fft_output[(offset << 1) + 1], 2))
            if d > max_val:
                max_val = d
                deltax = i
                deltay = j
            offset += 1

    if deltax > width >> 1:
        deltax -= width
    if deltay > height >> 1:
        deltay -= height

    # clean:
    # fft_input1, fft_input2, fft_output are automatically garbage collected (equivalent to free)

    return (ret, deltax, deltay)


def main() -> int:
    """
    Main function demonstrating phase correlation shift detection.

    Creates two test images with known offset and computes the shift
    between them using phase correlation.

    Returns:
        0 on success, 1 on failure
    """
    image1 = [0] * (256 * 128)
    image2 = [0] * (256 * 128)

    # Generate pair of images
    for j in range(128):
        for i in range(256):
            offset = i + j * 256
            if (i >= 16) and (i < 76) and (j >= 32) and (j < 92):
                image1[offset] = 128
            else:
                image1[offset] = 0

            if (i >= 8) and (i < 68) and (j >= 40) and (j < 100):
                image2[offset] = 16
            else:
                image2[offset] = 255

    ret, deltax, deltay = compute_shift(image1, image2, 256, 128)
    if ret:
        print("Operation failed", file=sys.stderr)
        return 1

    print(f"Calculated shift: [{deltax}, {deltay}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
