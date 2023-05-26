import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# generate encoding matrix
def make_encode(height, width):
    while True:
        # Random matrix height^2, width^2 in interval (0, 2)
        encoded = 2 * np.random.rand(height**2, width**2)
        ret, encoded = cv2.threshold(encoded, 0.5, 1, cv2.THRESH_BINARY)

        if cv2.determinant(encoded) != 0:
            break
    return encoded


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


encoded = make_encode(height, width)
plt.figure()
plt.imshow(encoded, cmap="gray")
plt.savefig(f"random.png")
plt.close()

errors = []
for i in range(height * width):
    print(i)
    tank = encoded[0 : i + 1, :]  # (1,hw)→(2,hw),・・・(hw,hw)
    mask_inv = np.linalg.pinv(tank)  # (hw,1)→(hw,2)・・・(hw,hw)
    out = np.dot(tank, img_array)  # (i+1, hw) * (hw, 1) = (i+1, 1)
    reconstruct = np.dot(mask_inv, out)  # (hw, i+1) * (i+1, 1) = (hw, 1)
    reimg = reconstruct.reshape(height, width).astype("uint8")

    if i % 100 == 0 or i == height * width - 1:
        plt.figure()
        plt.imshow(reimg, cmap="gray")
        plt.savefig(f"reconstruct_{i}.png")
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
plt.savefig("errors.png")


print("Done")
