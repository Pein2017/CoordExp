from PIL import Image

from probes.training_set_completion import coco22_review_packets as review


def test_single_bbox_has_full_original_overlay_and_labeled_context(tmp_path):
    image = Image.new("RGB", (320, 240), "white")
    original_box = [150, 100, 180, 130]
    full = tmp_path / "individual.png"
    context = tmp_path / "context.png"
    views = review.annotated_bbox_views(image, original_box, "gt:123:0", full, context)
    assert views["original_bbox_pixel_xyxy"] == original_box
    assert views["individual_full_image_bbox_overlay"] == review.binding(full)
    assert views["context_crop_with_bbox_and_id"] == review.binding(context)
    with Image.open(full) as overlay:
        assert overlay.size == image.size
        assert overlay.getpixel((150, 100)) == (0, 110, 230)
    with Image.open(context) as crop:
        local = views["bbox_pixel_xyxy_within_context"]
        assert crop.getpixel((local[0], local[1])) == (255, 80, 0)
        assert crop.getpixel((5, 5)) != (255, 255, 255)


def test_invalid_order_stays_visible_as_cross_in_context(tmp_path):
    view = review.annotated_bbox_views(
        Image.new("RGB", (320, 240), "white"), [220, 140, 180, 100],
        "proposal:invalid:0", tmp_path / "full.png", tmp_path / "context.png")
    assert view["original_bbox_pixel_xyxy"] == [220, 140, 180, 100]
    assert view["individual_full_image_bbox_overlay"]["path"] == str(tmp_path / "full.png")
    assert view["context_crop_with_bbox_and_id"]["path"] == str(tmp_path / "context.png")
