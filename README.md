# ComfyUI-bitpoet-IG4Inpaint

ComfyUI conditioning node for reference-conditioned Ideogram 4 LoRAs trained
with the `ideogram4_reference_conditioning_v1` sequence contract.

This enables inpainting with the Ideogram 4 model.

## Status

This is an alpha release. Use at your own risk.

## Requirements

This node requires the accompanying native ComfyUI Ideogram 4 changes that pack:

```text
text | noisy output | clean reference
```

The reference uses indicator `4`, relative MRoPE time coordinate `1`, and an
Ideogram model timestep of `1.0`. Only output-token predictions are returned to
the sampler.

## Node

`Ideogram 4 Reference Conditioning` accepts:

- Positive Ideogram 4 conditioning
- One reference image
- The Flux 2 / Ideogram 4 VAE
- The target `Empty Flux 2 Latent`
- `stretch` or `center_crop` resizing

It returns the updated positive conditioning, the encoded reference latent, and
the resized reference preview.

Connect only its positive-conditioning output to `Dual Model CFG Guider`. Leave
the unconditional Ideogram 4 model and negative conditioning unchanged.

`stretch` is the default because it matches the current ai-toolkit training
preprocessing.

## Example workflow

An example is bundled at:

```text
workflows/ideogram4_reference_workflow.json
```

Copy `workflows/idoinpaint_reference.png` into ComfyUI's `input` folder, or
select another image in the workflow's `Load Image` node.
