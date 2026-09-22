import unittest

import torch

from NeuroOperators import SpectralAttentionOperator2d, SpectralKernelAttention2d


class SpectralKernelAttentionTests(unittest.TestCase):
    def test_all_frequency_sources_preserve_shape_and_gradients(self):
        for kind in ("fourier_low", "chebyshev_low", "fourier_high"):
            with self.subTest(kind=kind):
                torch.manual_seed(13)
                layer = SpectralKernelAttention2d(
                    channels=4,
                    rank=3,
                    spectral_kind=kind,
                    modes=4,
                )
                state = torch.randn(2, 4, 17, 19, requires_grad=True)
                conditioning = torch.randn(2, 4, 17, 19, requires_grad=True)
                result = layer(state, conditioning)
                result.square().mean().backward()
                self.assertEqual(result.shape, state.shape)
                self.assertGreater(float(conditioning.grad.abs().sum()), 0.0)
                self.assertGreater(float(layer.query.weight.grad.abs().sum()), 0.0)
                self.assertEqual(layer.summary()["spectral_kind"], kind)

    def test_high_frequency_source_rejects_constant_field(self):
        layer = SpectralKernelAttention2d(
            channels=3,
            rank=2,
            spectral_kind="fourier_high",
            modes=4,
        )
        state = torch.ones(1, 3, 16, 20)
        layer(state, torch.zeros_like(state))
        self.assertLess(layer.summary()["source_rms"], 1e-5)

    def test_operator_contains_exactly_one_attention_layer(self):
        model = SpectralAttentionOperator2d(
            modes=4,
            width=6,
            depth=3,
            input_features=3,
            output_features=1,
            attention_kind="fourier_low",
            attention_rank=4,
        )
        layers = [module for module in model.modules() if isinstance(module, SpectralKernelAttention2d)]
        output = model(torch.randn(2, 3, 21, 17))
        self.assertEqual(len(layers), 1)
        self.assertEqual(output.shape, (2, 1, 21, 17))


if __name__ == "__main__":
    unittest.main()
