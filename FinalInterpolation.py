import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_bspline_params(n, k):
    control_points = np.linspace(0, n, n)
    num_knots = n + k
    knots = np.zeros(num_knots)
    for i in range(num_knots):
        if i < k:
            knots[i] = 0
        elif k <= i <= n:
            knots[i] = i - k + 1
        elif i > n:
            knots[i] = n - k + 2

    return control_points, knots

def bspline_basis(t, i, k, knots):
    if k == 0:
        if knots[i] <= t < knots[i+1]:
            return 1
        else:
            return 0
    else:
        w1 = 0 if knots[i+k] == knots[i] else (t - knots[i]) / (knots[i+k] - knots[i])
        w2 = 0 if knots[i+k+1] == knots[i+1] else (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1])
        return w1 * bspline_basis(t, i, k-1, knots) + w2 * bspline_basis(t, i+1, k-1, knots)

def bspline_interpolation(img, scale_factor, k):
    h, w = img.shape
    new_h = int(scale_factor * h)
    new_w = int(scale_factor * w)
    n = max(new_h, new_w)
    control_points, knots = get_bspline_params(n-1, k+1)
    out = np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            x, y = j / scale_factor, i / scale_factor
            u, v = x + k, y + k
            sum = 0
            for m in range(k+1):
                for n in range(k+1):
                    if v-k+n >= 0 and v-k+n < h and u-k+m >= 0 and u-k+m < w:
                        sum += bspline_basis(u-m, np.searchsorted(knots, u)-k-1, k, knots) * bspline_basis(v-n, np.searchsorted(knots, v)-k-1, k, knots) * img[int(v-k+n), int(u-k+m)]
            out[i, j] = np.clip(sum, 0, 255)

    return out

# Get the input image from the user
filename = "electricals.png"
img = Image.open(filename).convert('L')
print("Input image:", img)
image_array = np.array(img)
print("Input image array shape:", image_array.shape)

# Perform linear B-spline interpolation on the image with a scale factor of 2
interpolated_image = bspline_interpolation(image_array, 2, 1)
print("Interpolated image array shape:", interpolated_image.shape)

# Save the interpolated image
interpolated_img = Image.fromarray(interpolated_image)
interpolated_img.save('finale.png')

# Display the original and interpolated images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_array, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')
ax[1].imshow(interpolated_image, cmap='gray')
ax[1].set_title("Interpolated Image")
ax[1].axis('off')
plt.show()
