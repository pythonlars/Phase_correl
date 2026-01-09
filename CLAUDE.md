# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python implementations of the Phase Correlation algorithm for calculating shift between two images. The implementations are direct translations from C/C++ code, maintaining exact functional equivalence with the original implementations.

**Key Algorithm:** Phase Correlation based image alignment using 2D FFT (Radix-2 Cooley-Tukey Decimation-in-Time approach).

## Agent Usage Guide

This project has 5 specialized agents available. Use them for the following scenarios:

### code-translator (opus)
**Primary use case for this project.**

Use when you need to:
- Translate new C/C++ image processing functions to Python
- Back-translate Python to C/C++ for performance validation
- Port algorithms to other languages (Rust, Go, etc.)
- Maintain strict functional equivalence across language implementations
- Convert complex number handling between C-style arrays and Python complex types

**Example tasks:**
- "Translate this new FFT optimization from C to Python"
- "Port the phase correlation algorithm to Rust"
- "Convert this NumPy-based implementation back to C-style interleaved arrays"

### comprehensive-tester (sonnet)
**Use after ANY code changes to core algorithms.**

Use when you need to:
- Verify algorithm modifications produce correct results
- Test cross-version equivalence between `phase_correl_c.py` and `phase_correl_cpp.py`
- Add new test cases for edge cases or new features
- Manually verify shift detection accuracy with real images
- Validate that both implementations still produce identical numerical results
- Test new image format loaders

**Example tasks:**
- "Test the modified FFT implementation thoroughly"
- "Add test cases for the new JPEG loader"
- "Verify that both C-style and C++-style implementations still produce identical results after the optimization"

### backend-architect (opus)
**Use when planning algorithmic or architectural changes.**

Use when you need to:
- Design performance optimizations for FFT or phase correlation
- Plan integration with NumPy/SciPy while maintaining pure Python fallback
- Architect support for non-power-of-2 image dimensions
- Design multi-threaded or GPU-accelerated processing
- Plan API design for using this as a library
- Optimize memory usage for large image processing
- Design caching strategies for repeated FFT operations

**Example tasks:**
- "Design an architecture for supporting both power-of-2 and arbitrary-sized images"
- "Plan how to add GPU acceleration using CuPy while maintaining CPU fallback"
- "Architect a batch processing system for phase correlation on multiple image pairs"

### research-synthesizer (opus)
**Use when you need to understand mathematical/algorithmic foundations.**

Use when you need to:
- Research alternative FFT algorithms (Bluestein's, mixed-radix, FFTW approaches)
- Understand mathematical theory behind phase correlation
- Investigate sub-pixel accuracy techniques
- Compare different image alignment algorithms
- Find optimization techniques from academic papers
- Research numerical stability improvements

**Example tasks:**
- "Research FFT algorithms that work on non-power-of-2 sizes"
- "Find papers on sub-pixel phase correlation accuracy"
- "Compare phase correlation vs feature-based image alignment methods"
- "Research how to improve phase correlation robustness to noise"

### database-engineer (opus)
**Rarely needed for this project.**

This agent is less relevant since the project doesn't use databases. Consider using only if:
- Building a web service that stores image alignment results in a database
- Creating a benchmark results database
- Designing storage for large-scale image processing pipelines

## Running Code

### Execute Implementations
```bash
# Run C-style implementation (demonstrates with PNG images)
python3 phase_correl_c.py

# Run C++-style implementation (demonstrates with generated test images)
python3 phase_correl_cpp.py

# Run comprehensive test suite
python3 test_equivalence.py
```

### Testing
The test suite validates all core functions and cross-version equivalence:
```bash
python3 test_equivalence.py
# Expected: 14/14 tests passing (100%)
```

## Architecture

### Two Implementation Styles

**1. C-Style (`phase_correl_c.py`)**
- Function-based procedural API
- Uses NumPy arrays and `np.fft.fft2()` / `np.fft.ifft2()` for FFT operations
- Complex numbers handled natively by NumPy (complex128 dtype)
- Vectorized operations for peak finding and array manipulations
- Return values as tuples: `(ret_code, deltax, deltay)` where ret_code is 0 on success, -1 on error
- Includes PNG loading with filter reversal (Paeth, Sub, Up, Average)
- Optional PIL/Pillow support for broader image format compatibility

**2. C++-Style (`phase_correl_cpp.py`)**
- Object-oriented API
- Uses NumPy arrays internally (np.uint8 dtype for images, np.complex128 for FFT)
- Uses `np.fft.fft2()` / `np.fft.ifft2()` for FFT operations
- `GrayscaleImage` class for image representation with vectorized drawing methods
- `PhaseCorrelation` static-only class (cannot be instantiated - raises RuntimeError)
- Exception-based error handling (raises `RuntimeError`)
- No image loading (uses generated test patterns)

### Core Algorithm Components

**FFT Pipeline (NumPy-optimized):**
- `np.fft.fft2()` - 2D forward FFT (replaces manual Radix-2 Cooley-Tukey implementation)
- `np.fft.ifft2()` - 2D inverse FFT

**Note:** The manual FFT functions (`gen2fftorder`, `radix2fft`, `sum2fft`, `dit2fft`, `fft2d`) have been replaced with NumPy's optimized FFT operations, providing 10-100x performance improvement.

**Phase Correlation:**
1. Convert images to complex arrays (real part = pixel value, imaginary part = 0)
2. Apply 2D FFT to both images
3. Compute normalized cross-power spectrum: `e^(i*phase_diff)` with unit magnitude
4. Apply inverse 2D FFT
5. Find peak in result matrix → peak location = (deltax, deltay) shift

### Important Constraints

- **Image dimensions MUST be powers of 2** (e.g., 64, 128, 256, 512)
- Non-power-of-2 dimensions will fail with error code -1 (C-style) or RuntimeError (C++-style)
- Both images must have identical dimensions
- The C-style version automatically pads PNG images to power-of-2 dimensions
- Shift wrapping: values > (dimension/2) are wrapped to negative (e.g., shift of 250 in 256-wide image = -6)

### PNG Loading (C-style only)

The `phase_correl_c.py` implementation includes PNG loading:
- `load_png_grayscale()` - Parses PNG using only stdlib (struct, zlib)
- Supports non-interlaced 8-bit RGBA PNGs
- Converts to grayscale: `0.299*R + 0.587*G + 0.114*B`
- `pad_image()` - Pads to power-of-2 dimensions with black (0) pixels
- Implements PNG filter reversal: None, Sub, Up, Average, Paeth
- `paeth_predictor()` - PNG Paeth filter predictor implementation

## File Modifications

### When Editing Core Algorithms

**CRITICAL:** Both implementations must maintain exact functional equivalence. If you modify the algorithm in one file, you MUST update the corresponding function in the other:

| C-style (`phase_correl_c.py`) | C++-style (`phase_correl_cpp.py`) |
|--------------------------------|-----------------------------------|
| `radix2fft()` | `PhaseCorrelation._radix2_fft()` |
| `sum2fft()` | `PhaseCorrelation._sum2_fft()` |
| `gen2fftorder()` | `PhaseCorrelation._gen_input_order()` |
| `dit2fft()` | `PhaseCorrelation._dit2_fft()` |
| `fft2d()` | `PhaseCorrelation._fft_2d()` |
| `comp_norm_cross_correlation()` | `PhaseCorrelation._comp_norm_cross_correlation()` |
| `compute_shift()` | `PhaseCorrelation.compute_shift()` |

### Complex Number Handling

Both implementations now use NumPy's native complex128 dtype:
```python
# Convert images to complex arrays
img1 = np.array(image1, dtype=np.complex128).reshape(height, width)
img2 = np.array(image2, dtype=np.complex128).reshape(height, width)

# Perform FFT
fft1 = np.fft.fft2(img1)
fft2 = np.fft.fft2(img2)

# Vectorized complex operations
cross_power = fft1 * np.conj(fft2)
normalized = np.exp(1j * np.angle(cross_power))
```

### Testing After Changes

**MANDATORY:** After any algorithmic changes, run the full test suite:
```bash
python3 test_equivalence.py
```

All 14 tests must pass (100% success rate). The test suite validates:
- **Individual components:** Bit-reversal order, butterfly operations, twiddle factors
- **FFT correctness:** Forward FFT, inverse FFT, DC component, frequency bins
- **2D FFT:** Various image sizes (64×64, 128×64, 256×128)
- **Phase correlation:** Shift detection with known offsets, wrapping behavior
- **Cross-version equivalence:** Both implementations produce identical (deltax, deltay) results
- **Edge cases:** Non-power-of-2 rejection, zero size rejection, mismatched dimensions

If tests fail, both versions must be fixed to restore equivalence.

## Dependencies

**Required:**
- `numpy` - Optimized FFT operations (`np.fft.fft2`, `np.fft.ifft2`), array operations
- `math` - Trigonometric functions (for PNG loading only)
- `sys` - Exit codes, stderr
- `struct` - Binary data parsing for PNG chunks
- `zlib` - PNG IDAT decompression
- `typing` - Type hints (List, Tuple, Optional)

**Optional:**
- `PIL/Pillow` - Enhanced image loading (JPEG, BMP, etc.) with fallback to pure Python PNG parser

**Minimum Python version:** 3.6 (for type hints)

## Common Tasks

### Add support for new image formats (JPEG, BMP, etc.)
1. Add loader function to `phase_correl_c.py` following `load_png_grayscale()` pattern
2. Return tuple: `(width: int, height: int, pixels: List[int])`
3. Ensure grayscale conversion (0-255 integer values)
4. Handle padding to power-of-2 dimensions
5. Add test cases to `test_equivalence.py`

### Performance

**Current implementation** uses NumPy's optimized FFT operations:
- Both files now use `np.fft.fft2()` and `np.fft.ifft2()`
- Provides **10-100x speedup** over manual FFT implementation
- Uses vectorized NumPy operations for peak finding and array manipulations
- Performance comparable to or better than optimized C implementation when using NumPy with MKL

**Further optimization options:**
- Use `scipy.fft.fft2` for potential additional speed (marginal improvement)
- Consider `pyfftw` (Python wrapper for FFTW) for best performance
- Profile with `cProfile` to identify remaining bottlenecks

**IMPORTANT:** All optimizations maintain numerical equivalence (verified by test suite).

### Support non-power-of-2 dimensions
**Requires major architectural change:**
- Current Radix-2 FFT only works on power-of-2 sizes
- Options:
  1. Implement Bluestein's algorithm (works for any size)
  2. Implement mixed-radix FFT (works for composite sizes)
  3. Pad to next power-of-2 (already implemented)
  4. Use SciPy FFT (handles any size)

**Recommendation:** Use **backend-architect** agent to design the approach, then **code-translator** to implement in both versions.

### Add sub-pixel accuracy
Use **research-synthesizer** agent to find papers on sub-pixel phase correlation techniques (e.g., peak interpolation, Gaussian fitting).

## Documentation References

- `README.md` - User-facing documentation, usage examples, algorithm overview
- `CONVERSION_REPORT.md` - Detailed conversion methodology, type mappings, test results, performance analysis
- Original paper: Kuglin & Hines (1975) "The Phase Correlation Image Alignment Method"
- Image files: `img1.png`, `img2.png` - Test images for demonstration
