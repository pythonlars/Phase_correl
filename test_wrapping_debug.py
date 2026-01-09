#!/usr/bin/env python3
"""Debug test for shift wrapping behavior."""

import phase_correl_c

# Test wrapping behavior
width, height = 256, 256

# Image1: Rectangle at (20, 20)
image1 = [0] * (width * height)
for y in range(20, 60):
    for x in range(20, 60):
        image1[x + y * width] = 150

# Image2: Rectangle at (220, 20) - this is a shift of 200 pixels to the right
image2 = [0] * (width * height)
for y in range(20, 60):
    for x in range(220, 260):
        if x < width:
            image2[x + y * width] = 150

ret, dx, dy = phase_correl_c.compute_shift(image1, image2, width, height)

print(f"Test: Rectangle at (20,20) in img1, at (220,20) in img2")
print(f"Direct shift would be: (220-20, 0) = (200, 0)")
print(f"Since 200 > 128 (width/2), expected wrapped shift: (200-256, 0) = (-56, 0)")
print(f"Actual detected shift: ({dx}, {dy})")
print()

# The phase correlation detects shift FROM image1 TO image2
# So if image2's rectangle is at x=220 and image1's is at x=20,
# that means image2 is shifted by (220-20)=200 pixels to the right
# Phase correlation should detect -200 pixels (to align image2 back to image1)
# Since -200 < -128, it wraps to -200+256 = 56

print("Alternative interpretation:")
print("Phase correlation finds the shift TO APPLY to image2 to align with image1")
print("To move rectangle from x=220 to x=20: need to shift LEFT by 200")
print("Shift left by 200 = shift right by -200 = shift right by 56 (wrapped)")
print()

# Let's test the other way
print("\n" + "="*70)
print("Reverse test: Rectangle at (220,20) in img1, at (20,20) in img2")

image1_rev = [0] * (width * height)
for y in range(20, 60):
    for x in range(220, 260):
        if x < width:
            image1_rev[x + y * width] = 150

image2_rev = [0] * (width * height)
for y in range(20, 60):
    for x in range(20, 60):
        image2_rev[x + y * width] = 150

ret, dx, dy = phase_correl_c.compute_shift(image1_rev, image2_rev, width, height)
print(f"Actual detected shift: ({dx}, {dy})")
