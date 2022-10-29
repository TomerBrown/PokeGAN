import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

import dataset

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    """ Given a list of images, show them in a line. """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_transformation():
    """ Get the transformation to apply to the images. """
    return transforms.Compose([
        transforms.Lambda(dataset.pad_to_square),
        transforms.Resize((64, 64)),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25, fill=(255, 255, 255)),
        transforms.ColorJitter(contrast=0.2, hue=0.2, saturation=0.2, brightness=0.2),
        transforms.ToTensor()
    ]
    )
