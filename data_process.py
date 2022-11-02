import torch
import torchvision
import numpy as np

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

x_data = np.zeros((10000, 784))
label_data = np.zeros((10000, 10))
cnt = 0
for img, label in loader:
    x_data[cnt] = np.array(img.reshape(784,), dtype=np.float32)
    label_data[cnt] = np.array([1 if i == label[0] else 0 for i in range(10)], dtype=np.float32)
    cnt += 1

print(cnt)
print(x_data.shape, label_data.shape)

with open("fanshionMNIST_data_x", "wb") as fp:
    x_data.astype(np.float32).tofile(fp)

with open("fanshionMNIST_data_label", "wb") as fp:
    label_data.astype(np.float32).tofile(fp)