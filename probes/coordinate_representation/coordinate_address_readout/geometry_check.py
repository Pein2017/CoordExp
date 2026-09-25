"""Exercise installed processor patch grouping and vision positional ordering."""
from types import SimpleNamespace
import inspect
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from src.artifacts.utf8_json import binding


def check(processor):
    image_processor = processor.image_processor
    patch = image_processor.patch_size
    merge = image_processor.merge_size
    assert merge == 2
    images = []
    expected = []
    for offset, (hm, wm) in zip((0, 10), ((2, 3), (3, 2)), strict=True):
        cells = np.arange(hm*wm).reshape(hm, wm) + offset
        pixels = np.repeat(np.repeat(cells, patch*merge, 0), patch*merge, 1)
        images.append(Image.fromarray(np.repeat(pixels[..., None], 3, 2).astype('uint8')))
        expected.extend(cells.reshape(-1).tolist())
    encoded = image_processor(images=images, do_resize=False, do_rescale=False,
                              do_normalize=False, return_tensors='pt')
    observed = encoded['pixel_values'].float().reshape(-1, merge*merge,
                                               encoded['pixel_values'].shape[-1]).mean((1,2))
    wanted = torch.tensor(expected, dtype=observed.dtype)
    assert torch.equal(observed, wanted), 'processor merge order or image boundary differs'
    caught = False
    try:
        assert torch.equal(observed, wanted.roll(1)), 'deliberate visual order corruption'
    except AssertionError:
        caught = True
    assert caught
    # Execute the installed spatial-position implementation with an identity frequency table.
    fake = SimpleNamespace(spatial_merge_size=2,
                           rotary_pos_emb=lambda size: torch.arange(size)[:, None])
    positions = Qwen3VLVisionModel.rot_pos_emb(fake, torch.tensor([[1,4,6],[1,6,4]]))
    centers = positions.reshape(-1,4,2).float().mean(1)
    expected_centers = torch.tensor([[2*r+.5, 2*c+.5] for h,w in ((2,3),(3,2))
                                     for r in range(h) for c in range(w)])
    assert torch.equal(centers, expected_centers), 'vision ordering differs from pixel grouping'
    return dict(status='pass', rectangular_merged_grids=[[2,3],[3,2]],
                processor_image_boundary_identity=True, corrupted_order_rejected=caught,
                processor_source=binding(inspect.getfile(type(image_processor))),
                vision_source=binding(inspect.getfile(Qwen3VLVisionModel)))
