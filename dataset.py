import os.path as path

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import utils


def read_image_with_white_background(image_path):
    """ Given an image path, read the image and replace the alpha channel with a white background. """
    image = Image.open(image_path).convert('RGBA')
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    return torchvision.transforms.PILToTensor()(new_image.convert('RGB'))


def pad_to_square(image):
    """ Given an image, white-pad it to make it square. """
    c, h, w = image.shape
    dim_diff = abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    image = torch.nn.functional.pad(image, pad, "constant", 255)
    return image


class PokemonDataset(Dataset):
    def __init__(self, dir_path: str, transform=None):
        pokemon_csv_path = path.join(dir_path, "pokemon.csv")
        data = pd.read_csv(pokemon_csv_path, encoding='utf-8')
        self.images = data['path'].values
        self.images = [path.join(dir_path, 'data', image) for image in self.images]
        self.labels = data['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image_with_white_background(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_num_square_images(self):
        square_count = 0
        for image, label in tqdm(self):
            if image.shape[1] == image.shape[2]:
                square_count += 1
        return square_count

    def get_largest_image_shape(self):

        largest_shape = (-1, 1)
        for image, _ in tqdm(self):
            if image.shape[1] * image.shape[2] > largest_shape[0] * largest_shape[1]:
                largest_shape = (image.shape[1], image.shape[2])
        return largest_shape

    def get_images_statistics(self):
        """ Returns some basic statistics about the images in the dataset to allow further decisions. """
        n_square_images = self.get_num_square_images()
        n_images = len(self)
        square_images_ratio = n_square_images / n_images
        largest_shape = self.get_largest_image_shape()
        mean, std = self.get_mean_and_std()

        return {
            "n_images": n_images,
            "n_square_images": n_square_images,
            "square_images_ratio": square_images_ratio,
            "largest_shape": largest_shape,
            "mean": mean,
            "std": std
        }

    def get_mean_and_std(self):
        mean, std = 0, 0
        for image, _ in tqdm(self):
            mean += image.float().mean(axis=(1, 2))
            std += image.float().std(axis=(1, 2))
        return mean / len(self), std / len(self)


def get_data_loader(batch_size=1, shuffle=True, num_workers=1, dir_path: str = '.',
                    transform=utils.get_transformation()):
    poke_dataset = PokemonDataset(dir_path, transform)
    return DataLoader(
        dataset=poke_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
