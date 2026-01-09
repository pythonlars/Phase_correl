"""
Phase Correlation implementation translated from C++ to Python.
Provides exact functional equivalence to the original phase_correl.cpp.
Uses NumPy for efficient FFT and array operations.
"""

import numpy as np
from numpy.typing import NDArray
import sys
from typing import List, Tuple


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
            self._data = _source._data.copy()
            self._height, self._width = self._data.shape
        else:
            # Normal constructor: allocate and fill with specified value
            self._data = np.full((height, width), fill, dtype=np.uint8)
            self._width = width
            self._height = height

    @classmethod
    def from_copy(cls, source: 'GrayscaleImage') -> 'GrayscaleImage':
        """Copy constructor equivalent."""
        return cls(_source=source)

    def copy_from(self, source: 'GrayscaleImage') -> 'GrayscaleImage':
        """Assignment operator equivalent."""
        self._data = source._data.copy()
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
        x_end = min(x + width, self._width)
        y_end = min(y + height, self._height)
        if x < x_end and y < y_end:
            self._data[y:y_end, x:x_end] = fill

    def get_width(self) -> int:
        """Get image width."""
        return self._width

    def get_height(self) -> int:
        """Get image height."""
        return self._height

    def get_data(self) -> List[int]:
        """Get image data as a list of pixel values (row-major order)."""
        return self._data.flatten().tolist()

    def get_array(self) -> NDArray[np.uint8]:
        """Get image data as a NumPy array."""
        return self._data

    def set(self, width: int, height: int, data: List[int]) -> None:
        """
        Set image data.

        Args:
            width: New image width
            height: New image height
            data: New pixel data (row-major order)
        """
        self._data = np.array(data, dtype=np.uint8).reshape((height, width))
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

        # Get image data as complex arrays
        img1 = image1.get_array().astype(np.complex128)
        img2 = image2.get_array().astype(np.complex128)

        # Perform 2D FFT on each image
        fft1 = np.fft.fft2(img1)
        fft2 = np.fft.fft2(img2)

        # Compute normalized cross power spectrum
        # cross_power = fft1 * conj(fft2)
        # normalized = exp(i * angle(cross_power))
        cross_power = fft1 * np.conj(fft2)
        normalized = np.exp(1j * np.angle(cross_power))

        # Perform inverse 2D FFT on obtained matrix
        correlation = np.fft.ifft2(normalized)

        # Search for peak using magnitude
        magnitude = np.abs(correlation)
        peak_idx = np.argmax(magnitude)
        deltay, deltax = np.unravel_index(peak_idx, magnitude.shape)

        # Wrap around for negative shifts
        if deltax > (width >> 1):
            deltax -= width
        if deltay > (height >> 1):
            deltay -= height

        return (int(deltax), int(deltay))


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
