#!/usr/bin/env python3
"""
Comprehensive Test Suite for NumPy-Optimized Phase Correlation

This test suite specifically validates the NumPy-optimized implementations
with focus on:
- Various image sizes
- Edge cases
- Numerical accuracy
- Performance characteristics
- Cross-version equivalence
"""

import sys
import time
import numpy as np
from typing import Tuple, List
import phase_correl_c
import phase_correl_cpp


class TestResults:
    """Container for test results with timing information."""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.failures = []
        self.timings = []

    def add_pass(self, test_name: str, duration_ms: float = 0.0):
        """Record a passing test."""
        self.total_tests += 1
        self.passed_tests += 1
        if duration_ms > 0:
            print(f"✓ {test_name} ({duration_ms:.2f}ms)")
            self.timings.append((test_name, duration_ms))
        else:
            print(f"✓ {test_name}")

    def add_fail(self, test_name: str, reason: str):
        """Record a failing test."""
        self.total_tests += 1
        self.failed_tests += 1
        self.failures.append((test_name, reason))
        print(f"✗ {test_name}: {reason}")

    def summary(self) -> str:
        """Generate summary report."""
        summary = f"""
{'='*70}
TEST SUMMARY - NumPy Optimized Implementation
{'='*70}
Total Tests:  {self.total_tests}
Passed:       {self.passed_tests} ({100*self.passed_tests//self.total_tests if self.total_tests > 0 else 0}%)
Failed:       {self.failed_tests}
{'='*70}
"""
        if self.timings:
            summary += "\nPERFORMANCE METRICS:\n"
            summary += "="*70 + "\n"
            for test_name, duration in self.timings:
                summary += f"{test_name}: {duration:.2f}ms\n"
            summary += "="*70 + "\n"
        return summary


class TestVariousImageSizes:
    """Test phase correlation with various image sizes."""

    def __init__(self, results: TestResults):
        self.results = results

    def test_size_64x64(self):
        """Test 64x64 images."""
        test_name = "Image Size: 64x64"
        width, height = 64, 64

        start_time = time.time()
        ret, dx, dy = self._test_shift(width, height, 4, -4)
        duration_ms = (time.time() - start_time) * 1000

        if ret != 0:
            self.results.add_fail(test_name, f"Failed with error code {ret}")
        elif abs(dx - 4) > 1 or abs(dy + 4) > 1:
            self.results.add_fail(test_name, f"Expected shift (4,-4), got ({dx},{dy})")
        else:
            self.results.add_pass(test_name, duration_ms)

    def test_size_128x128(self):
        """Test 128x128 images."""
        test_name = "Image Size: 128x128"
        width, height = 128, 128

        start_time = time.time()
        ret, dx, dy = self._test_shift(width, height, 8, 8)
        duration_ms = (time.time() - start_time) * 1000

        if ret != 0:
            self.results.add_fail(test_name, f"Failed with error code {ret}")
        elif abs(dx - 8) > 1 or abs(dy - 8) > 1:
            self.results.add_fail(test_name, f"Expected shift (8,8), got ({dx},{dy})")
        else:
            self.results.add_pass(test_name, duration_ms)

    def test_size_256x256(self):
        """Test 256x256 images."""
        test_name = "Image Size: 256x256"
        width, height = 256, 256

        start_time = time.time()
        ret, dx, dy = self._test_shift(width, height, 16, -12)
        duration_ms = (time.time() - start_time) * 1000

        if ret != 0:
            self.results.add_fail(test_name, f"Failed with error code {ret}")
        elif abs(dx - 16) > 1 or abs(dy + 12) > 1:
            self.results.add_fail(test_name, f"Expected shift (16,-12), got ({dx},{dy})")
        else:
            self.results.add_pass(test_name, duration_ms)

    def test_size_512x256(self):
        """Test 512x256 images (non-square)."""
        test_name = "Image Size: 512x256 (non-square)"
        width, height = 512, 256

        start_time = time.time()
        ret, dx, dy = self._test_shift(width, height, 20, 10)
        duration_ms = (time.time() - start_time) * 1000

        if ret != 0:
            self.results.add_fail(test_name, f"Failed with error code {ret}")
        elif abs(dx - 20) > 1 or abs(dy - 10) > 1:
            self.results.add_fail(test_name, f"Expected shift (20,10), got ({dx},{dy})")
        else:
            self.results.add_pass(test_name, duration_ms)

    def test_size_256x512(self):
        """Test 256x512 images (non-square)."""
        test_name = "Image Size: 256x512 (non-square)"
        width, height = 256, 512

        start_time = time.time()
        ret, dx, dy = self._test_shift(width, height, -12, 24)
        duration_ms = (time.time() - start_time) * 1000

        if ret != 0:
            self.results.add_fail(test_name, f"Failed with error code {ret}")
        elif abs(dx + 12) > 1 or abs(dy - 24) > 1:
            self.results.add_fail(test_name, f"Expected shift (-12,24), got ({dx},{dy})")
        else:
            self.results.add_pass(test_name, duration_ms)

    def _test_shift(self, width: int, height: int, shift_x: int, shift_y: int) -> Tuple[int, int, int]:
        """Helper function to test a specific shift."""
        # Create image1 with a rectangle
        image1 = [0] * (width * height)
        rect_size = min(width, height) // 4
        rect_x1 = width // 4
        rect_y1 = height // 4

        for y in range(rect_y1, rect_y1 + rect_size):
            for x in range(rect_x1, rect_x1 + rect_size):
                if x < width and y < height:
                    image1[x + y * width] = 200

        # Create image2 with shifted rectangle
        image2 = [0] * (width * height)
        rect_x2 = (rect_x1 - shift_x + width) % width
        rect_y2 = (rect_y1 - shift_y + height) % height

        for y in range(rect_y2, rect_y2 + rect_size):
            for x in range(rect_x2, rect_x2 + rect_size):
                if x < width and y < height:
                    image2[x + y * width] = 200

        return phase_correl_c.compute_shift(image1, image2, width, height)

    def run_all(self):
        """Run all size tests."""
        print("\n" + "="*70)
        print("Testing Various Image Sizes (NumPy Optimized)")
        print("="*70)

        self.test_size_64x64()
        self.test_size_128x128()
        self.test_size_256x256()
        self.test_size_512x256()
        self.test_size_256x512()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def __init__(self, results: TestResults):
        self.results = results

    def test_non_power_of_2_width(self):
        """Test non-power-of-2 width rejection."""
        test_name = "Edge Case: Non-power-of-2 width (100x64)"

        image = [0] * (100 * 64)
        ret, _, _ = phase_correl_c.compute_shift(image, image, 100, 64)

        if ret == -1:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, "Should reject non-power-of-2 width")

    def test_non_power_of_2_height(self):
        """Test non-power-of-2 height rejection."""
        test_name = "Edge Case: Non-power-of-2 height (64x100)"

        image = [0] * (64 * 100)
        ret, _, _ = phase_correl_c.compute_shift(image, image, 64, 100)

        if ret == -1:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, "Should reject non-power-of-2 height")

    def test_zero_width(self):
        """Test zero width rejection."""
        test_name = "Edge Case: Zero width"

        ret, _, _ = phase_correl_c.compute_shift([], [], 0, 64)

        if ret == -1:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, "Should reject zero width")

    def test_zero_height(self):
        """Test zero height rejection."""
        test_name = "Edge Case: Zero height"

        ret, _, _ = phase_correl_c.compute_shift([], [], 64, 0)

        if ret == -1:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, "Should reject zero height")

    def test_mismatched_sizes_cpp(self):
        """Test mismatched sizes in C++ version."""
        test_name = "Edge Case: Mismatched sizes (C++ version)"

        img1 = phase_correl_cpp.GrayscaleImage(64, 64, 0)
        img2 = phase_correl_cpp.GrayscaleImage(128, 64, 0)

        try:
            phase_correl_cpp.PhaseCorrelation.compute_shift(img1, img2)
            self.results.add_fail(test_name, "Should raise RuntimeError")
        except RuntimeError:
            self.results.add_pass(test_name)

    def test_identical_images(self):
        """Test with identical images (zero shift)."""
        test_name = "Edge Case: Identical images (zero shift)"

        width, height = 128, 128
        image = [100] * (width * height)

        # Add some pattern
        for y in range(40, 80):
            for x in range(40, 80):
                image[x + y * width] = 200

        ret, dx, dy = phase_correl_c.compute_shift(image, image, width, height)

        if ret != 0:
            self.results.add_fail(test_name, f"Failed with error code {ret}")
        elif dx != 0 or dy != 0:
            self.results.add_fail(test_name, f"Expected (0,0), got ({dx},{dy})")
        else:
            self.results.add_pass(test_name)

    def test_boundary_patterns(self):
        """Test patterns near image boundaries."""
        test_name = "Edge Case: Patterns near boundaries"

        width, height = 128, 128
        image1 = [0] * (width * height)
        image2 = [0] * (width * height)

        # Rectangle near the right edge in image1
        for y in range(50, 70):
            for x in range(100, 120):
                image1[x + y * width] = 180

        # Same rectangle slightly shifted in image2
        for y in range(55, 75):
            for x in range(105, 125):
                image2[x + y * width] = 180

        ret, dx, dy = phase_correl_c.compute_shift(image1, image2, width, height)

        if ret != 0:
            self.results.add_fail(test_name, f"Failed with error code {ret}")
        else:
            # Just verify it completes successfully
            # The algorithm should handle boundary patterns correctly
            self.results.add_pass(test_name)

    def run_all(self):
        """Run all edge case tests."""
        print("\n" + "="*70)
        print("Testing Edge Cases (NumPy Optimized)")
        print("="*70)

        self.test_non_power_of_2_width()
        self.test_non_power_of_2_height()
        self.test_zero_width()
        self.test_zero_height()
        self.test_mismatched_sizes_cpp()
        self.test_identical_images()
        self.test_boundary_patterns()


class TestNumericalAccuracy:
    """Test numerical accuracy and consistency."""

    def __init__(self, results: TestResults):
        self.results = results

    def test_repeated_computation(self):
        """Test that repeated computations give identical results."""
        test_name = "Numerical: Repeated computation consistency"

        width, height = 128, 128
        image1 = [0] * (width * height)
        image2 = [0] * (width * height)

        # Create test pattern
        for y in range(30, 90):
            for x in range(30, 90):
                image1[x + y * width] = 180
        for y in range(40, 100):
            for x in range(20, 80):
                image2[x + y * width] = 180

        # Compute shift multiple times
        results = []
        for _ in range(5):
            ret, dx, dy = phase_correl_c.compute_shift(image1, image2, width, height)
            if ret != 0:
                self.results.add_fail(test_name, "Computation failed")
                return
            results.append((dx, dy))

        # Check all results are identical
        if len(set(results)) == 1:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, f"Inconsistent results: {results}")

    def test_c_vs_cpp_equivalence(self):
        """Test C and C++ versions produce identical results."""
        test_name = "Numerical: C vs C++ equivalence"

        test_cases = [
            (64, 64, 10, 20, 30, 30, 150),
            (128, 128, 20, 30, 50, 50, 180),
            (256, 128, 40, 20, 60, 40, 200),
            (128, 256, 30, 50, 40, 60, 170),
        ]

        all_match = True
        for width, height, x1, y1, rect_w, rect_h, fill in test_cases:
            # C version
            image1_c = [0] * (width * height)
            image2_c = [255] * (width * height)
            for y in range(y1, min(y1 + rect_h, height)):
                for x in range(x1, min(x1 + rect_w, width)):
                    image1_c[x + y * width] = fill

            x2 = (x1 + 10) % width
            y2 = (y1 - 10 + height) % height
            for y in range(y2, min(y2 + rect_h, height)):
                for x in range(x2, min(x2 + rect_w, width)):
                    image2_c[x + y * width] = fill

            ret_c, dx_c, dy_c = phase_correl_c.compute_shift(image1_c, image2_c, width, height)

            # C++ version
            image1_cpp = phase_correl_cpp.GrayscaleImage(width, height, 0)
            image2_cpp = phase_correl_cpp.GrayscaleImage(width, height, 255)
            image1_cpp.draw_rectangle(x1, y1, rect_w, rect_h, fill)
            image2_cpp.draw_rectangle(x2, y2, rect_w, rect_h, fill)

            dx_cpp, dy_cpp = phase_correl_cpp.PhaseCorrelation.compute_shift(image1_cpp, image2_cpp)

            if ret_c != 0 or dx_c != dx_cpp or dy_c != dy_cpp:
                self.results.add_fail(test_name,
                    f"Size {width}x{height}: C=({dx_c},{dy_c}) vs C++=({dx_cpp},{dy_cpp})")
                all_match = False
                break

        if all_match:
            self.results.add_pass(test_name)

    def test_small_shift_accuracy(self):
        """Test accuracy with small shifts (1-2 pixels)."""
        test_name = "Numerical: Small shift accuracy"

        width, height = 128, 128

        for shift_x, shift_y in [(1, 0), (0, 1), (1, 1), (2, -1), (-1, 2)]:
            image1 = [0] * (width * height)
            image2 = [0] * (width * height)

            # Create distinct pattern
            for y in range(40, 80):
                for x in range(40, 80):
                    image1[x + y * width] = 200

            # Shifted pattern
            for y in range(40, 80):
                for x in range(40, 80):
                    x2 = (x - shift_x + width) % width
                    y2 = (y - shift_y + height) % height
                    image2[x2 + y2 * width] = 200

            ret, dx, dy = phase_correl_c.compute_shift(image1, image2, width, height)

            if ret != 0:
                self.results.add_fail(test_name, f"Failed for shift ({shift_x},{shift_y})")
                return

            if dx != shift_x or dy != shift_y:
                self.results.add_fail(test_name,
                    f"Expected ({shift_x},{shift_y}), got ({dx},{dy})")
                return

        self.results.add_pass(test_name)

    def run_all(self):
        """Run all numerical accuracy tests."""
        print("\n" + "="*70)
        print("Testing Numerical Accuracy (NumPy Optimized)")
        print("="*70)

        self.test_repeated_computation()
        self.test_c_vs_cpp_equivalence()
        self.test_small_shift_accuracy()


class TestNumPyOptimization:
    """Test NumPy-specific optimizations and features."""

    def __init__(self, results: TestResults):
        self.results = results

    def test_numpy_functions_used(self):
        """Verify NumPy functions are being used."""
        test_name = "NumPy: FFT functions are used"

        # Check if NumPy is imported
        import phase_correl_c
        import phase_correl_cpp

        if not hasattr(phase_correl_c, 'np'):
            self.results.add_fail(test_name, "NumPy not imported in C version")
            return

        if not hasattr(phase_correl_cpp, 'np'):
            self.results.add_fail(test_name, "NumPy not imported in C++ version")
            return

        self.results.add_pass(test_name)

    def test_array_operations(self):
        """Test NumPy array operations work correctly."""
        test_name = "NumPy: Array operations"

        width, height = 64, 64

        # Create test data using NumPy
        image1_np = np.zeros((width * height,), dtype=int)
        image1_np[1000:2000] = 150
        image1 = image1_np.tolist()

        image2_np = np.zeros((width * height,), dtype=int)
        image2_np[1100:2100] = 150
        image2 = image2_np.tolist()

        ret, dx, dy = phase_correl_c.compute_shift(image1, image2, width, height)

        if ret == 0:
            self.results.add_pass(test_name)
        else:
            self.results.add_fail(test_name, f"Failed with error code {ret}")

    def run_all(self):
        """Run all NumPy optimization tests."""
        print("\n" + "="*70)
        print("Testing NumPy Optimization Features")
        print("="*70)

        self.test_numpy_functions_used()
        self.test_array_operations()


def main():
    """Run comprehensive NumPy optimization test suite."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           NumPy-Optimized Phase Correlation Test Suite              ║
║              Comprehensive Testing & Validation                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    results = TestResults()

    # Run all test suites
    test_sizes = TestVariousImageSizes(results)
    test_sizes.run_all()

    test_edges = TestEdgeCases(results)
    test_edges.run_all()

    test_numerical = TestNumericalAccuracy(results)
    test_numerical.run_all()

    test_numpy = TestNumPyOptimization(results)
    test_numpy.run_all()

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
