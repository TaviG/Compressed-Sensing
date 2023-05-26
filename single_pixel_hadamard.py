import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_hadamard(shape):
    # Check if it is 2 raised to the nth power
    n = np.log(shape) / np.log(2)
    if 2**n == shape:
        print("making hadamard")
    else:
        print("error")
        return None

    hadamard_shape = [2**i for i in range(int(n) + 1)]

    hadamard = np.array([1])
    hadamard_matrix = {1: hadamard}
    iteration = len(hadamard_shape) - 1
    for i in range(iteration):
        hadamard = np.hstack((hadamard, hadamard))
        hadamard = np.vstack((hadamard, hadamard))
        # If the index to flip is (4,4), flip the last (2,2)
        reverse_range = -hadamard_shape[i]
        hadamard[reverse_range:, reverse_range:] = (
            hadamard[reverse_range:, reverse_range:] * -1
        )
        hadamard_matrix[hadamard_shape[i + 1]] = hadamard

    return hadamard_matrix[shape]


# root mean square error
def RMSE(img1, img2):
    n = len(img1)
    dif = img1 - img2
    dif2 = dif**2
    rmse = np.sqrt(np.sum(dif2) / (n))
    return rmse


img = Image.open("lena.png")
img = np.array(img.resize((32, 32)))
# img = np.array(img)

height, width = img.shape
img_array = img.reshape(height * width, 1)

# Hadamard mask making
hadamard_matrix = make_hadamard(height * width)

plt.figure()
plt.imshow(hadamard_matrix, cmap="gray")
plt.savefig("hadamard.png")

errors = []
for i in range(height * width):
    print(i)
    tank = hadamard_matrix[0 : i + 1, :]  # (1,hw)→(2,hw),・・・(hw,hw)
    mask_inv = np.linalg.pinv(tank)  # (hw,1)→(hw,2)・・・(hw,hw)
    out = np.dot(tank, img_array)  # (i+1, hw) * (hw, 1) = (i+1, 1)
    reconstruct = np.dot(mask_inv, out)  # (hw, i+1) * (i+1, 1) = (hw, 1)
    reimg = reconstruct.reshape(height, width).astype("uint8")

    if i % 100 == 0 or i == height * width - 1:
        plt.figure()
        plt.imshow(reimg, cmap="gray")
        plt.savefig(f"reconstruct_hadamard_{i}.png")
        plt.close()

    error = RMSE(img_array, reconstruct)
    errors.append(error)

plt.figure(figsize=(12, 8))
plt.ylabel("RMSE", fontsize=25)
plt.xlabel("iteration number", fontsize=25)
plt.ylim(0, max(errors) * 1.1)
plt.tick_params(labelsize=20)
plt.grid(which="major", color="black", linestyle="-")
plt.plot(np.arange(1, height * width + 1), errors)
plt.savefig("errors_hadamard.png")
