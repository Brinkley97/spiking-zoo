import torch, torchvision

import numpy as np

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def build_cifar_ten_pytorch(show_data: bool = False):
    """Load the CIFAR-10 dataset via PyTorch (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=mnist)
    
    Parameters:
    -----------
    show_data: `bool`
        Show example images
        Call the `imshow()` function
    """
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)


    if show_data == True:
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



