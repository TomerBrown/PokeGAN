import unittest

import torch

from model import Discriminator, Generator, initialize_weights


class TestPokemon(unittest.TestCase):
    def test_model_shapes(self):
        N, in_channels, H, W = 8, 3, 64, 64
        z_dim = 100
        x = torch.randn((N, in_channels, H, W))
        disc = Discriminator(in_channels, 8)
        initialize_weights(disc)
        self.assertTrue(disc(x).shape == (N, 1, 1, 1)), "Discriminator test failed"
        gen = Generator(z_dim, in_channels, 8)
        initialize_weights(gen)
        z = torch.randn((N, z_dim, 1, 1))
        self.assertTrue(gen(z).shape == (N, in_channels, H, W)), "Generator test failed"


if __name__ == '__main__':
    unittest.main()
