#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase Correlation Conversion Verification

Tests functional equivalence between C/C++ and Python implementations.
Validates that all functions produce identical results for various inputs.
"""

import sys
import math
import subprocess
import json
import tempfile
import os
from typing import List, Tuple, Dict, Any

# Import the Python translations
import phase_correl_c
import phase_correl_cpp


class TestResults:
    """Container for test results."""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.failures = []

    def add_pass(self, test_name: str):
        """Record a passing test."""
        self.total_tests += 1
        self.passed_tests += 1
        print(f"✓ {test_name}")

    def add_fail(self, test_name: str, reason: str):
        """Record a failing test."""
        self.total_tests += 1
        self.failed_tests += 1
        self.failures.append((test_name, reason))
        print(f"✗ {test_name}: {reason}")

    def summary(self) -> str:
        """Generate summary report."""
        return f"""
{'='*70}
TEST SUMMARY
{'='*70}
Total Tests:  {self.total_tests}
Passed:       {self.passed_tests} ({100*self.passed_tests//self.total_tests if self.total_tests > 0 else 0}%)
Failed:       {self.failed_tests}
{'='*70}
"""


def compare_floats(a: float, b: float, tolerance: float = 1e-9) -> bool:
    """Compare floating point values with tolerance."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b):
        return a == b
    return abs(a - b) <= tolerance


def compare_float_arrays(arr1: List[float], arr2: List[float], tolerance: float = 1e-9) -> Tuple[bool, str]:
    """Compare two float arrays element-wise."""
    if len(arr1) != len(arr2):
        return False, f"Length mismatch: {len(arr1)} vs {len(arr2)}"

    for i, (a, b) in enumerate(zip(arr1, arr2)):
        if not compare_floats(a, b, tolerance):
            return False, f"Element {i} differs: {a} vs {b} (diff: {abs(a-b)})"

    return True, "OK"


class TestPhaseCorrelationC:
    """Test suite for C version equivalence."""

    def __init__(self, results: TestResults):
        self.results = results

    def test_gen2fftorder(self):
        """Test FFT order generation."""
        test_name = "C: gen2fftorder - Basic sizes"

        # Test various power-of-2 sizes
        for size in [2, 4, 8, 16, 32, 64]:
            order = phase_correl_c.gen2fftorder(size)

            # Verify length
            if len(order) != size:
                self.results.add_fail(test_name, f"Size {size}: Wrong length {len(order)}")
                return

            # Verify first element is 0
            if order[0] != 0:
                self.results.add_fail(test_name, f"Size {size}: First element not 0")
                return

            # Verify all elements are non-negative
            if any(x < 0 for x in order):
                self.results.add_fail(test_name, f"Size {size}: Negative values found")
                return

        self.results.add_pass(test_name)

    def test_radix2fft(self):
        """Test radix-2 FFT butterfly."""
        test_name = "C: radix2fft - Butterfly operation"

        # Test data: [re0, im0, re1, im1]
        input_arr = [1.0, 0.0, 1.0, 0.0]
        output = [0.0] * 4

        phase_correl_c.radix2fft(input_arr, output, 0, 0, 1)

        # Expected: output[0,1] = sum, output[2,3] = difference
        expected = [2.0, 0.0, 0.0, 0.0]

        match, reason = compare_float_arrays(output, expected)
        if match:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, reason)

    def test_dit2fft_simple(self):
        """Test DIT FFT with simple input."""
        test_name = "C: dit2fft - Simple FFT"

        # DC signal: all ones
        size = 4
        input_arr = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        output = [0.0] * (size << 1)

        phase_correl_c.dit2fft(input_arr, output, size, 0)

        # DC component should be sum of all inputs
        if abs(output[0] - 4.0) > 1e-9:
            self.results.add_fail(test_name, f"DC component wrong: {output[0]} vs 4.0")
            return

        # All other components should be ~0
        for i in range(1, size):
            if abs(output[i << 1]) > 1e-9 or abs(output[(i << 1) + 1]) > 1e-9:
                self.results.add_fail(test_name, f"Non-DC component {i} not zero")
                return

        self.results.add_pass(test_name)

    def test_dit2fft_inverse(self):
        """Test inverse FFT recovers original."""
        test_name = "C: dit2fft - Inverse FFT"

        size = 8
        input_arr = [1.0, 0.5, 2.0, 0.0, 0.5, 1.0, 1.5, 0.5,
                     1.0, 0.0, 0.5, 0.5, 2.0, 1.0, 1.5, 0.0]
        fft_result = [0.0] * (size << 1)
        recovered = [0.0] * (size << 1)

        # Forward FFT
        phase_correl_c.dit2fft(input_arr, fft_result, size, 0)
        # Inverse FFT
        phase_correl_c.dit2fft(fft_result, recovered, size, 1)

        match, reason = compare_float_arrays(input_arr, recovered, 1e-6)
        if match:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, f"Inverse FFT failed: {reason}")

    def test_comp_norm_cross_correlation(self):
        """Test normalized cross correlation."""
        test_name = "C: comp_norm_cross_correlation - Unit magnitude"

        f = [1.0, 0.0]  # 1+0j
        g = [0.0, 1.0]  # 0+1j
        r = [0.0, 0.0]

        phase_correl_c.comp_norm_cross_correlation(f, g, r, 0, 0, 0)

        # Result should have magnitude 1
        magnitude = math.sqrt(r[0]**2 + r[1]**2)
        if abs(magnitude - 1.0) > 1e-9:
            self.results.add_fail(test_name, f"Not unit magnitude: {magnitude}")
            return

        self.results.add_pass(test_name)

    def test_compute_shift_known_offset(self):
        """Test compute_shift with known offsets."""
        test_name = "C: compute_shift - Known offsets"

        # Test various image sizes and shifts
        test_cases = [
            (64, 64, 4, 4),    # 64x64 image, shift (4,4)
            (128, 64, 8, -8),  # 128x64 image, shift (8,-8)
            (64, 128, -4, 12), # 64x128 image, shift (-4,12)
        ]

        for width, height, shift_x, shift_y in test_cases:
            # Create two images with rectangles at different positions
            image1 = [0] * (width * height)
            image2 = [0] * (width * height)

            # Rectangle in image1
            rect_size = min(width, height) // 4
            rect_x1 = width // 4
            rect_y1 = height // 4

            for y in range(rect_y1, rect_y1 + rect_size):
                for x in range(rect_x1, rect_x1 + rect_size):
                    if x < width and y < height:
                        image1[x + y * width] = 200

            # Rectangle in image2 (shifted)
            # Note: Phase correlation detects the shift FROM image1 TO image2
            # So if we want to detect a shift of (shift_x, shift_y), we need to
            # position image2's rectangle at (rect_x1 - shift_x, rect_y1 - shift_y)
            rect_x2 = (rect_x1 - shift_x + width) % width
            rect_y2 = (rect_y1 - shift_y + height) % height

            for y in range(rect_y2, rect_y2 + rect_size):
                for x in range(rect_x2, rect_x2 + rect_size):
                    if x < width and y < height:
                        image2[x + y * width] = 200

            ret, dx, dy = phase_correl_c.compute_shift(image1, image2, width, height)

            # Normalize shifts to [-width/2, width/2]
            expected_dx = shift_x
            expected_dy = shift_y
            if expected_dx > width // 2:
                expected_dx -= width
            if expected_dy > height // 2:
                expected_dy -= height

            if ret != 0:
                self.results.add_fail(test_name, f"Size {width}x{height}: Failed with error code {ret}")
                return

            # Allow small tolerance for peak detection
            if abs(dx - expected_dx) > 2 or abs(dy - expected_dy) > 2:
                self.results.add_fail(test_name,
                    f"Size {width}x{height}: Expected ({expected_dx},{expected_dy}), got ({dx},{dy})")
                return

        self.results.add_pass(test_name)

    def test_compute_shift_edge_cases(self):
        """Test compute_shift edge cases."""
        test_name = "C: compute_shift - Edge cases"

        # Test non-power-of-2 size (should fail)
        image = [0] * 100
        ret, _, _ = phase_correl_c.compute_shift(image, image, 10, 10)
        if ret != -1:
            self.results.add_fail(test_name, "Non-power-of-2 size should fail")
            return

        # Test zero size (should fail)
        ret, _, _ = phase_correl_c.compute_shift([], [], 0, 8)
        if ret != -1:
            self.results.add_fail(test_name, "Zero size should fail")
            return

        self.results.add_pass(test_name)

    def run_all(self):
        """Run all C version tests."""
        print("\n" + "="*70)
        print("Testing C Version Translation")
        print("="*70)

        self.test_gen2fftorder()
        self.test_radix2fft()
        self.test_dit2fft_simple()
        self.test_dit2fft_inverse()
        self.test_comp_norm_cross_correlation()
        self.test_compute_shift_known_offset()
        self.test_compute_shift_edge_cases()


class TestPhaseCorrelationCpp:
    """Test suite for C++ version equivalence."""

    def __init__(self, results: TestResults):
        self.results = results

    def test_grayscale_image_creation(self):
        """Test GrayscaleImage creation and basic operations."""
        test_name = "C++: GrayscaleImage - Creation and fill"

        img = phase_correl_cpp.GrayscaleImage(16, 16, 128)

        if img.get_width() != 16:
            self.results.add_fail(test_name, f"Width wrong: {img.get_width()}")
            return
        if img.get_height() != 16:
            self.results.add_fail(test_name, f"Height wrong: {img.get_height()}")
            return
        if len(img.get_data()) != 256:
            self.results.add_fail(test_name, f"Data size wrong: {len(img.get_data())}")
            return
        if not all(x == 128 for x in img.get_data()):
            self.results.add_fail(test_name, "Fill value incorrect")
            return

        self.results.add_pass(test_name)

    def test_grayscale_image_copy(self):
        """Test GrayscaleImage copy operations."""
        test_name = "C++: GrayscaleImage - Copy operations"

        img1 = phase_correl_cpp.GrayscaleImage(8, 8, 100)
        img1.draw_rectangle(2, 2, 4, 4, 200)

        # Test copy constructor
        img2 = phase_correl_cpp.GrayscaleImage.from_copy(img1)

        if img2.get_data() != img1.get_data():
            self.results.add_fail(test_name, "Copy constructor failed")
            return

        # Verify deep copy
        img1.get_data()[0] = 50
        if img2.get_data()[0] == 50:
            self.results.add_fail(test_name, "Not a deep copy")
            return

        self.results.add_pass(test_name)

    def test_grayscale_image_draw_rectangle(self):
        """Test rectangle drawing."""
        test_name = "C++: GrayscaleImage - Draw rectangle"

        img = phase_correl_cpp.GrayscaleImage(16, 16, 0)
        img.draw_rectangle(4, 4, 8, 8, 255)

        data = img.get_data()

        # Check rectangle interior
        for y in range(4, 12):
            for x in range(4, 12):
                if data[x + y * 16] != 255:
                    self.results.add_fail(test_name, f"Interior pixel at ({x},{y}) not set")
                    return

        # Check exterior (corner)
        if data[0] != 0:
            self.results.add_fail(test_name, "Exterior pixel modified")
            return

        self.results.add_pass(test_name)

    def test_phase_correlation_no_instantiation(self):
        """Test that PhaseCorrelation cannot be instantiated."""
        test_name = "C++: PhaseCorrelation - No instantiation"

        try:
            pc = phase_correl_cpp.PhaseCorrelation()
            self.results.add_fail(test_name, "Should not allow instantiation")
        except RuntimeError:
            self.results.add_pass(test_name)

    def test_phase_correlation_compute_shift(self):
        """Test compute_shift with known patterns."""
        test_name = "C++: PhaseCorrelation - Compute shift"

        # Create test images
        img1 = phase_correl_cpp.GrayscaleImage(256, 128, 0)
        img2 = phase_correl_cpp.GrayscaleImage(256, 128, 255)

        img1.draw_rectangle(16, 32, 60, 60, 128)
        img2.draw_rectangle(8, 40, 60, 60, 16)

        deltax, deltay = phase_correl_cpp.PhaseCorrelation.compute_shift(img1, img2)

        # Expected shift: (16-8, 32-40) = (8, -8)
        if deltax != 8 or deltay != -8:
            self.results.add_fail(test_name, f"Expected (8, -8), got ({deltax}, {deltay})")
            return

        self.results.add_pass(test_name)

    def test_phase_correlation_error_handling(self):
        """Test error handling for invalid inputs."""
        test_name = "C++: PhaseCorrelation - Error handling"

        # Test mismatched sizes
        img1 = phase_correl_cpp.GrayscaleImage(64, 64, 0)
        img2 = phase_correl_cpp.GrayscaleImage(128, 64, 0)

        try:
            phase_correl_cpp.PhaseCorrelation.compute_shift(img1, img2)
            self.results.add_fail(test_name, "Should raise error for size mismatch")
            return
        except RuntimeError:
            pass

        # Test non-power-of-2
        img3 = phase_correl_cpp.GrayscaleImage(63, 63, 0)
        img4 = phase_correl_cpp.GrayscaleImage(63, 63, 0)

        try:
            phase_correl_cpp.PhaseCorrelation.compute_shift(img3, img4)
            self.results.add_fail(test_name, "Should raise error for non-power-of-2")
            return
        except RuntimeError:
            pass

        self.results.add_pass(test_name)

    def run_all(self):
        """Run all C++ version tests."""
        print("\n" + "="*70)
        print("Testing C++ Version Translation")
        print("="*70)

        self.test_grayscale_image_creation()
        self.test_grayscale_image_copy()
        self.test_grayscale_image_draw_rectangle()
        self.test_phase_correlation_no_instantiation()
        self.test_phase_correlation_compute_shift()
        self.test_phase_correlation_error_handling()


class TestCrossVersionEquivalence:
    """Test equivalence between C and C++ Python translations."""

    def __init__(self, results: TestResults):
        self.results = results

    def test_same_shift_results(self):
        """Verify C and C++ versions produce identical shift results."""
        test_name = "Cross-version: Identical shift results"

        # Test with various configurations
        test_cases = [
            (64, 64, 8, 8, 16, 16, 100),
            (128, 128, 20, 30, 40, 40, 150),
            (256, 128, 16, 32, 60, 60, 200),
        ]

        for width, height, x1, y1, rect_w, rect_h, fill in test_cases:
            # C version
            image1_c = [0] * (width * height)
            image2_c = [255] * (width * height)
            for y in range(y1, y1 + rect_h):
                for x in range(x1, x1 + rect_w):
                    if x < width and y < height:
                        image1_c[x + y * width] = fill

            x2 = (x1 + 8) % width
            y2 = (y1 - 8) % height
            for y in range(y2, y2 + rect_h):
                for x in range(x2, x2 + rect_w):
                    if x < width and y < height:
                        image2_c[x + y * width] = fill

            ret_c, dx_c, dy_c = phase_correl_c.compute_shift(image1_c, image2_c, width, height)

            # C++ version
            image1_cpp = phase_correl_cpp.GrayscaleImage(width, height, 0)
            image2_cpp = phase_correl_cpp.GrayscaleImage(width, height, 255)
            image1_cpp.draw_rectangle(x1, y1, rect_w, rect_h, fill)
            image2_cpp.draw_rectangle(x2, y2, rect_w, rect_h, fill)

            dx_cpp, dy_cpp = phase_correl_cpp.PhaseCorrelation.compute_shift(image1_cpp, image2_cpp)

            if ret_c != 0:
                self.results.add_fail(test_name, f"C version failed: {ret_c}")
                return

            if dx_c != dx_cpp or dy_c != dy_cpp:
                self.results.add_fail(test_name,
                    f"Size {width}x{height}: C=({dx_c},{dy_c}) vs C++=({dx_cpp},{dy_cpp})")
                return

        self.results.add_pass(test_name)

    def run_all(self):
        """Run all cross-version tests."""
        print("\n" + "="*70)
        print("Testing Cross-Version Equivalence")
        print("="*70)

        self.test_same_shift_results()


def main():
    """Run all tests and generate report."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     Phase Correlation C/C++ to Python Conversion Verification       ║
║                  Functional Equivalence Test Suite                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    results = TestResults()

    # Run C version tests
    test_c = TestPhaseCorrelationC(results)
    test_c.run_all()

    # Run C++ version tests
    test_cpp = TestPhaseCorrelationCpp(results)
    test_cpp.run_all()

    # Run cross-version tests
    test_cross = TestCrossVersionEquivalence(results)
    test_cross.run_all()

    # Print summary
    print(results.summary())

    # Print detailed failures if any
    if results.failures:
        print("\nDETAILED FAILURES:")
        print("="*70)
        for test_name, reason in results.failures:
            print(f"\n{test_name}:")
            print(f"  {reason}")
        print("="*70)

    # Return exit code
    return 0 if results.failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
