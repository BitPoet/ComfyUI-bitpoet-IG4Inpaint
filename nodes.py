import comfy.utils
import node_helpers
from datetime import datetime
from typing import Tuple
from torch import Tensor
import math

class Ideogram4ReferenceConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "reference_image": ("IMAGE",),
                "vae": ("VAE",),
                "target_latent": ("LATENT",),
                "resize_mode": (
                    ["stretch", "center_crop"],
                    {
                        "default": "stretch",
                        "tooltip": "Stretch matches ai-toolkit reference training. Center crop preserves aspect ratio.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE")
    RETURN_NAMES = ("positive", "reference_latent", "resized_reference")
    FUNCTION = "apply"
    CATEGORY = "conditioning/ideogram4"
    DESCRIPTION = (
        "Encodes one reference image at the target size and attaches its latent "
        "only to Ideogram 4 positive conditioning."
    )

    def apply(self, positive, reference_image, vae, target_latent, resize_mode):
        target_samples = target_latent.get("samples")
        if target_samples is None or target_samples.ndim != 4:
            raise ValueError("Ideogram 4 target latent must contain a 4D samples tensor")
        if target_samples.shape[1] != 128:
            raise ValueError(
                "Ideogram 4 target latent must have 128 channels, got "
                f"{target_samples.shape[1]}"
            )

        target_height = target_samples.shape[-2] * 16
        target_width = target_samples.shape[-1] * 16
        crop = "center" if resize_mode == "center_crop" else "disabled"
        resized_reference = comfy.utils.common_upscale(
            reference_image.movedim(-1, 1),
            target_width,
            target_height,
            "bilinear",
            crop,
        ).movedim(1, -1)

        reference_samples = vae.encode(resized_reference)
        if reference_samples.ndim != 4:
            raise ValueError(
                "Ideogram 4 reference VAE output must be 4D, got "
                f"{tuple(reference_samples.shape)}"
            )
        if reference_samples.shape[1:] != target_samples.shape[1:]:
            raise ValueError(
                "Ideogram 4 reference latent must match the target latent channels "
                f"and spatial size, got {tuple(reference_samples.shape[1:])} and "
                f"{tuple(target_samples.shape[1:])}"
            )

        positive = node_helpers.conditioning_set_values(
            positive,
            {"reference_latents": [reference_samples]},
        )
        return (
            positive,
            {"samples": reference_samples},
            resized_reference,
        )

class BitpotDateString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("datestring",)
    FUNCTION = "getresult"
    CATEGORY = "BitPoet/helpers"
    DESCRIPTION = (
        "Output a date string in yyyyMMdd format for the current date"
    )

    def getresult(self):
        return( datetime.today().strftime('%Y%m%d') )


class BitPoetClosestMultiples:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ({"type": "INT", "default": 1024,}),
                "height": ({"type": "INT", "default": 1024,}),
            },
            "optional": {
                "base": ({"type": "INT", "default": 16, "min": 0, "max": 128, "defaultInput": False,}),
                "up": ({"type": "BOOLEAN", "default": True, "defaultInput": False,}),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "closest_multiples"
    CATEGORY = "BitPoet/helpers"
    DESCRIPTION = (
        "Calulates the closest multiples of the given base for the given width and height."
        "Rounds up or down depending on the 'Up' switch."
    )
    
    def closest_multiples(self, width, height, base, up):
        if base == 0:
            raise ValueError("base must not be 0")

        def round_value(n: int):
            if up:
                # Round up to next multiple
                return n if n % base == 0 else n + (base - n % base)
            else:
                # Round down to previous multiple
                return n - (n % base)

        return( round_value(width), round_value(height), )


class BitPoetImageToMultipleOf:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple_of": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "method": (["center crop", "rescale"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, image: Tensor, multiple_of: int, method: str) -> Tuple[Tensor]:
        """Center crop the image to a specific multiple of a number."""
        _, height, width, _ = image.shape

        new_height = height - (height % multiple_of)
        new_width = width - (width % multiple_of)

        if method == "rescale":
            return (
                F.interpolate(
                    image.unsqueeze(0),
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0),
            )
        else:
            top = (height - new_height) // 2
            left = (width - new_width) // 2
            bottom = top + new_height
            right = left + new_width
            return (image[:, top:bottom, left:right, :],)


NODE_CLASS_MAPPINGS = {
    "Ideogram4ReferenceConditioning": Ideogram4ReferenceConditioning,
    "BitpotDateString": BitpotDateString,
    "BitPoetClosestMultiples": BitPoetClosestMultiples,
    "BitPoetImageToMultipleOf": BitPoetImageToMultipleOf,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4ReferenceConditioning": "Ideogram 4 Reference Conditioning",
    "BitpotDateString": "BitPoet Date String Helper",
    "BitPoetClosestMultiples": "BitPoet Closest Multiple (w+h)",
    "BitPoetImageToMultipleOf": "Scale Image to Multiples of Base",
}
