import importlib.util
from pathlib import Path
import unittest

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("bitpoet_idoinpaint_nodes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Ideogram4ReferenceConditioning = MODULE.Ideogram4ReferenceConditioning


class FakeVAE:
    def encode(self, image):
        batch, height, width, _ = image.shape
        return torch.zeros(batch, 128, height // 16, width // 16)


class Ideogram4ReferenceConditioningTests(unittest.TestCase):
    def test_node_attaches_one_matching_reference_latent(self):
        positive = [[torch.zeros(1, 2, 3), {"existing": True}]]
        reference = torch.rand(1, 20, 30, 3)
        target = {"samples": torch.zeros(1, 128, 2, 3)}

        conditioned, reference_latent, resized = Ideogram4ReferenceConditioning().apply(
            positive,
            reference,
            FakeVAE(),
            target,
            "stretch",
        )

        self.assertEqual(resized.shape, (1, 32, 48, 3))
        self.assertEqual(reference_latent["samples"].shape, target["samples"].shape)
        self.assertIs(conditioned[0][1]["existing"], True)
        self.assertEqual(len(conditioned[0][1]["reference_latents"]), 1)
        self.assertEqual(conditioned[0][1]["reference_latents"][0].shape, target["samples"].shape)
        self.assertNotIn("reference_latents", positive[0][1])

    def test_node_rejects_non_ideogram_target_latent(self):
        target = {"samples": torch.zeros(1, 16, 2, 3)}

        with self.assertRaisesRegex(ValueError, "128 channels"):
            Ideogram4ReferenceConditioning().apply(
                [[None, {}]],
                torch.rand(1, 32, 48, 3),
                FakeVAE(),
                target,
                "stretch",
            )


if __name__ == "__main__":
    unittest.main()
