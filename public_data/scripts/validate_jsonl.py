#!/usr/bin/env python3
"""
Validate JSONL files for Qwen3-VL training.

Checks:
- JSON format validity
- Required fields presence
- Data types correctness
- Image file existence
- Geometry format and bounds (bbox_2d / poly)
- Coord-token values (<|coord_k|>, k in 0..999) when present
- Reject legacy/unsupported geometry keys (bbox, polygon, line, line_points)
- Summary statistics
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coord_tokens.codec import is_coord_token, token_to_int


DISALLOWED_GEOMETRY_KEYS = {"bbox", "polygon", "line", "line_points"}


def _as_float_coord(value: Any) -> Optional[float]:
    """Return numeric coordinate value for comparisons, supporting coord tokens."""
    if is_coord_token(value):
        try:
            return float(token_to_int(str(value)))
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sequence_has_coord_tokens(values: Sequence[Any]) -> bool:
    return any(is_coord_token(v) for v in values)


def _sequence_has_numbers(values: Sequence[Any]) -> bool:
    return any(isinstance(v, (int, float)) for v in values)


class ValidationError:
    """Container for validation errors."""
    def __init__(self, line_num: int, field: str, message: str):
        self.line_num = line_num
        self.field = field
        self.message = message
    
    def __str__(self):
        return f"Line {self.line_num}, field '{self.field}': {self.message}"


class JSONLValidator:
    """Validator for Qwen3-VL JSONL format."""
    
    def __init__(
        self,
        check_images: bool = True,
        verbose: bool = False,
        *,
        expected_max_pixels: Optional[int] = None,
        expected_multiple_of: Optional[int] = None,
        check_image_sizes: bool = False,
    ) -> None:
        self.check_images = check_images
        self.check_image_sizes = bool(check_image_sizes) and bool(check_images)
        self.expected_max_pixels = expected_max_pixels
        self.expected_multiple_of = expected_multiple_of
        self.verbose = verbose
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
        self.stats = {
            "total_lines": 0,
            "valid_samples": 0,
            "total_objects": 0,
            "missing_images": 0,
            "image_open_failed": 0,
            "image_size_mismatch": 0,
            "oversize_images": 0,
            "non_multiple_dims": 0,
            "invalid_bboxes": 0,
            "invalid_polys": 0,
            "invalid_geometries": 0,
            "categories": set(),
        }
    
    def validate_file(self, jsonl_path: str) -> bool:
        """
        Validate entire JSONL file.
        
        Returns:
            True if all samples valid
        """
        if not os.path.exists(jsonl_path):
            print(f"✗ File not found: {jsonl_path}")
            return False
        
        print(f"Validating: {jsonl_path}")
        print("="*60)
        
        jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path))
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                self.stats["total_lines"] += 1
                
                # Parse JSON
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    self.errors.append(ValidationError(
                        line_num, "json", f"Invalid JSON: {e}"
                    ))
                    continue
                
                # Validate sample
                if self.validate_sample(sample, line_num, jsonl_dir):
                    self.stats["valid_samples"] += 1
        
        return self.print_results()
    
    def validate_sample(
        self,
        sample: Dict[str, Any],
        line_num: int,
        base_dir: str,
    ) -> bool:
        """
        Validate one sample.

        Expected format:
        {
          "images": [str],
          "objects": [{"bbox_2d": [x1,y1,x2,y2], "desc": str}],
          "width": int,
          "height": int,
          "summary": str  # optional
        }

        Notes:
        - Objects may use either `bbox_2d` or `poly` geometry (exactly one).
        - Geometry coordinates may be pixel-space numbers or coord tokens (<|coord_k|>).

        Returns:
            True if sample is valid
        """
        sample_valid = True

        # Check required fields
        required_fields = ["images", "objects", "width", "height"]
        for field in required_fields:
            if field not in sample:
                self.errors.append(ValidationError(line_num, field, "Required field missing"))
                sample_valid = False

        if not sample_valid:
            return False

        image_abs: Optional[str] = None

        # Validate 'images' field
        images = sample["images"]
        if not isinstance(images, list):
            self.errors.append(
                ValidationError(line_num, "images", f"Must be list, got {type(images).__name__}")
            )
            sample_valid = False
        elif len(images) != 1:
            self.warnings.append(f"Line {line_num}: Expected 1 image, got {len(images)}")
        elif images:
            image_path = images[0]
            if not isinstance(image_path, str):
                self.errors.append(
                    ValidationError(
                        line_num,
                        "images[0]",
                        f"Must be string, got {type(image_path).__name__}",
                    )
                )
                sample_valid = False
            else:
                # Enforce the global contract: images MUST be relative to the JSONL directory.
                if os.path.isabs(image_path):
                    self.errors.append(
                        ValidationError(
                            line_num,
                            "images[0]",
                            "Image path must be relative to the JSONL directory (docs/data/JSONL_CONTRACT.md)",
                        )
                    )
                    sample_valid = False
                else:
                    image_abs = os.path.join(base_dir, image_path)
                    if self.check_images:
                        # Resolve relative path and ensure the file exists.
                        if not os.path.exists(image_abs):
                            self.errors.append(
                                ValidationError(
                                    line_num,
                                    "images[0]",
                                    f"Image not found: {image_abs}",
                                )
                            )
                            self.stats["missing_images"] += 1
                            sample_valid = False

        # Validate 'width' and 'height'
        width = sample["width"]
        height = sample["height"]
        width_ok = isinstance(width, int) and width > 0
        height_ok = isinstance(height, int) and height > 0
        if not width_ok:
            self.errors.append(
                ValidationError(line_num, "width", f"Must be positive int, got {width}")
            )
            sample_valid = False
        if not height_ok:
            self.errors.append(
                ValidationError(line_num, "height", f"Must be positive int, got {height}")
            )
            sample_valid = False

        if width_ok and height_ok:
            if self.expected_max_pixels is not None:
                max_pixels = int(self.expected_max_pixels)
                pixels = int(width) * int(height)
                if pixels > max_pixels:
                    self.errors.append(
                        ValidationError(
                            line_num,
                            "width,height",
                            f"Image pixels exceed max_pixels: {pixels} > {max_pixels}",
                        )
                    )
                    self.stats["oversize_images"] += 1
                    sample_valid = False

            if self.expected_multiple_of is not None:
                factor = int(self.expected_multiple_of)
                if factor > 0 and (width % factor != 0 or height % factor != 0):
                    self.errors.append(
                        ValidationError(
                            line_num,
                            "width,height",
                            f"Image dims must be multiples of {factor}, got {(width, height)}",
                        )
                    )
                    self.stats["non_multiple_dims"] += 1
                    sample_valid = False

            if self.check_image_sizes and image_abs is not None and os.path.exists(image_abs):
                try:
                    with Image.open(image_abs) as img:
                        actual_w, actual_h = img.size
                    if (actual_w, actual_h) != (width, height):
                        self.errors.append(
                            ValidationError(
                                line_num,
                                "images[0]",
                                f"Image size mismatch vs JSONL meta: file={(actual_w, actual_h)} meta={(width, height)}",
                            )
                        )
                        self.stats["image_size_mismatch"] += 1
                        sample_valid = False
                except Exception as e:
                    self.errors.append(
                        ValidationError(
                            line_num,
                            "images[0]",
                            f"Failed to open image for size check: {image_abs} ({type(e).__name__}: {e})",
                        )
                    )
                    self.stats["image_open_failed"] += 1
                    sample_valid = False

        # Validate 'objects' field
        objects = sample["objects"]
        if not isinstance(objects, list):
            self.errors.append(
                ValidationError(line_num, "objects", f"Must be list, got {type(objects).__name__}")
            )
            sample_valid = False
        else:
            self.stats["total_objects"] += len(objects)
            if width_ok and height_ok:
                for obj_idx, obj in enumerate(objects):
                    if not self.validate_object(obj, line_num, obj_idx, width, height):
                        sample_valid = False
            else:
                # If width/height are invalid, object-bound checks are meaningless.
                sample_valid = False

        # Validate optional 'summary'
        if "summary" in sample:
            if not isinstance(sample["summary"], str):
                self.warnings.append(
                    f"Line {line_num}: 'summary' should be string, got {type(sample['summary']).__name__}"
                )

        return sample_valid
    
    def validate_object(
        self,
        obj: Dict[str, Any],
        line_num: int,
        obj_idx: int,
        img_width: int,
        img_height: int,
    ) -> bool:
        """Validate one object annotation against docs/data/JSONL_CONTRACT.md."""
        prefix = f"objects[{obj_idx}]"

        if not isinstance(obj, dict):
            self.errors.append(
                ValidationError(
                    line_num,
                    prefix,
                    f"Must be object/dict, got {type(obj).__name__}",
                )
            )
            self.stats["invalid_geometries"] += 1
            return False

        obj_valid = True

        # Reject legacy/unsupported geometry keys (must be normalized by converters).
        for key in sorted(DISALLOWED_GEOMETRY_KEYS):
            if key in obj and obj.get(key) is not None:
                self.errors.append(
                    ValidationError(
                        line_num,
                        f"{prefix}.{key}",
                        "Legacy/unsupported geometry key; expected only bbox_2d or poly",
                    )
                )
                self.stats["invalid_geometries"] += 1
                obj_valid = False

        # Validate desc
        if "desc" not in obj:
            self.errors.append(
                ValidationError(line_num, f"{prefix}.desc", "Required field missing")
            )
            return False

        desc = obj.get("desc")
        if not isinstance(desc, str):
            self.errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}.desc",
                    f"Must be string, got {type(desc).__name__}",
                )
            )
            obj_valid = False
        elif not desc.strip():
            self.errors.append(
                ValidationError(line_num, f"{prefix}.desc", "Cannot be empty")
            )
            obj_valid = False
        else:
            self.stats["categories"].add(desc)

        # Exactly one geometry per object.
        has_bbox = ("bbox_2d" in obj) and (obj.get("bbox_2d") is not None)
        has_poly = ("poly" in obj) and (obj.get("poly") is not None)

        if has_bbox and has_poly:
            self.errors.append(
                ValidationError(
                    line_num,
                    prefix,
                    "Object has multiple geometry fields (bbox_2d + poly); exactly one is allowed",
                )
            )
            self.stats["invalid_geometries"] += 1
            return False

        if not has_bbox and not has_poly:
            self.errors.append(
                ValidationError(
                    line_num,
                    prefix,
                    "Missing geometry: expected bbox_2d or poly",
                )
            )
            self.stats["invalid_geometries"] += 1
            return False

        if "poly_points" in obj and not has_poly:
            self.errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}.poly_points",
                    "poly_points is only allowed when poly is present",
                )
            )
            self.stats["invalid_geometries"] += 1
            obj_valid = False

        if has_bbox:
            bbox = obj.get("bbox_2d")
            if not isinstance(bbox, list):
                self.errors.append(
                    ValidationError(
                        line_num,
                        f"{prefix}.bbox_2d",
                        f"Must be list, got {type(bbox).__name__}",
                    )
                )
                self.stats["invalid_bboxes"] += 1
                return False

            if len(bbox) != 4:
                self.errors.append(
                    ValidationError(
                        line_num,
                        f"{prefix}.bbox_2d",
                        f"Must have 4 values, got {len(bbox)}",
                    )
                )
                self.stats["invalid_bboxes"] += 1
                return False

            if _sequence_has_coord_tokens(bbox) and _sequence_has_numbers(bbox):
                self.warnings.append(
                    f"Line {line_num}, {prefix}.bbox_2d: Mixed pixel numbers and coord tokens in one geometry"
                )

            coords: List[float] = []
            for i, coord in enumerate(bbox):
                if is_coord_token(coord):
                    try:
                        coords.append(float(token_to_int(str(coord))))
                    except ValueError as exc:
                        self.errors.append(
                            ValidationError(
                                line_num,
                                f"{prefix}.bbox_2d[{i}]",
                                f"Invalid coord token: {exc}",
                            )
                        )
                        self.stats["invalid_bboxes"] += 1
                        obj_valid = False
                    continue

                if isinstance(coord, (int, float)):
                    coords.append(float(coord))
                    continue

                self.errors.append(
                    ValidationError(
                        line_num,
                        f"{prefix}.bbox_2d[{i}]",
                        f"Must be numeric or coord token, got {type(coord).__name__}",
                    )
                )
                self.stats["invalid_bboxes"] += 1
                obj_valid = False

            if len(coords) == 4:
                x1, y1, x2, y2 = coords

                if x2 <= x1:
                    self.errors.append(
                        ValidationError(
                            line_num,
                            f"{prefix}.bbox_2d",
                            f"Invalid: x2 ({x2}) <= x1 ({x1})",
                        )
                    )
                    self.stats["invalid_bboxes"] += 1
                    obj_valid = False

                if y2 <= y1:
                    self.errors.append(
                        ValidationError(
                            line_num,
                            f"{prefix}.bbox_2d",
                            f"Invalid: y2 ({y2}) <= y1 ({y1})",
                        )
                    )
                    self.stats["invalid_bboxes"] += 1
                    obj_valid = False

                # Bounds checks only make sense for pixel-space numbers.
                if not _sequence_has_coord_tokens(bbox):
                    if x2 < 0 or x1 > img_width:
                        self.warnings.append(
                            f"Line {line_num}, {prefix}.bbox_2d: Box completely outside image width"
                        )
                    if y2 < 0 or y1 > img_height:
                        self.warnings.append(
                            f"Line {line_num}, {prefix}.bbox_2d: Box completely outside image height"
                        )

            return obj_valid

        # poly
        poly = obj.get("poly")
        if not isinstance(poly, list):
            self.errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}.poly",
                    f"Must be list, got {type(poly).__name__}",
                )
            )
            self.stats["invalid_polys"] += 1
            return False

        if len(poly) < 6:
            self.errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}.poly",
                    f"Must have at least 6 values (>= 3 points), got {len(poly)}",
                )
            )
            self.stats["invalid_polys"] += 1
            obj_valid = False

        if len(poly) % 2 != 0:
            self.errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}.poly",
                    f"Must have even length (x,y pairs), got {len(poly)}",
                )
            )
            self.stats["invalid_polys"] += 1
            obj_valid = False

        if _sequence_has_coord_tokens(poly) and _sequence_has_numbers(poly):
            self.warnings.append(
                f"Line {line_num}, {prefix}.poly: Mixed pixel numbers and coord tokens in one geometry"
            )

        for i, coord in enumerate(poly):
            if is_coord_token(coord):
                try:
                    _ = token_to_int(str(coord))
                except ValueError as exc:
                    self.errors.append(
                        ValidationError(
                            line_num,
                            f"{prefix}.poly[{i}]",
                            f"Invalid coord token: {exc}",
                        )
                    )
                    self.stats["invalid_polys"] += 1
                    obj_valid = False
                continue

            if isinstance(coord, (int, float)):
                continue

            self.errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}.poly[{i}]",
                    f"Must be numeric or coord token, got {type(coord).__name__}",
                )
            )
            self.stats["invalid_polys"] += 1
            obj_valid = False

        poly_points = obj.get("poly_points")
        if poly_points is not None:
            if not isinstance(poly_points, int):
                self.errors.append(
                    ValidationError(
                        line_num,
                        f"{prefix}.poly_points",
                        f"Must be int when present, got {type(poly_points).__name__}",
                    )
                )
                self.stats["invalid_polys"] += 1
                obj_valid = False
            elif len(poly) % 2 == 0:
                expected = len(poly) // 2
                if poly_points != expected:
                    self.errors.append(
                        ValidationError(
                            line_num,
                            f"{prefix}.poly_points",
                            f"Expected {expected} (=len(poly)/2), got {poly_points}",
                        )
                    )
                    self.stats["invalid_polys"] += 1
                    obj_valid = False

        return obj_valid
    
    def print_results(self) -> bool:
        """Print validation results."""
        print("\n" + "="*60)
        print("Validation Results")
        print("="*60)
        
        # Statistics
        print(f"\n✓ Statistics:")
        print(f"  Total lines: {self.stats['total_lines']}")
        print(f"  Valid samples: {self.stats['valid_samples']}")
        print(f"  Total objects: {self.stats['total_objects']}")
        print(f"  Missing images: {self.stats['missing_images']}")
        print(f"  Invalid bboxes: {self.stats['invalid_bboxes']}")
        print(f"  Invalid polys: {self.stats['invalid_polys']}")
        print(f"  Unique categories: {len(self.stats['categories'])}")
        
        if self.stats['valid_samples'] > 0:
            avg_obj = self.stats['total_objects'] / self.stats['valid_samples']
            print(f"  Avg objects/sample: {avg_obj:.2f}")
        
        # Errors
        if self.errors:
            print(f"\n✗ Errors ({len(self.errors)}):")
            for error in self.errors[:20]:  # Show first 20
                print(f"  • {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")
        else:
            print("\n✓ No errors found")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  • {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        # Summary
        print("\n" + "="*60)
        if not self.errors:
            print("✓ VALIDATION PASSED")
            print("="*60 + "\n")
            return True
        else:
            print("✗ VALIDATION FAILED")
            print("="*60 + "\n")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSONL files for Qwen3-VL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected JSONL format:

{
  "images": ["path/to/image.jpg"],
  "objects": [
    {"bbox_2d": [x1, y1, x2, y2], "desc": "person"},
    {"poly": [x1, y1, x2, y2, x3, y3], "poly_points": 3, "desc": "car"}
  ],
  "width": 640,
  "height": 480,
  "summary": "A person standing next to a car"  # optional
}

Validation checks:
- JSON format correctness
- Required fields: images, objects, width, height
- Images list has exactly 1 element
- bbox_2d format: [x1, y1, x2, y2] with x2 > x1, y2 > y1
- poly format: flat [x1, y1, x2, y2, ...] with even length >= 6
- Coord-token values (<|coord_k|>, k in 0..999) are accepted for geometry coords
- Bboxes within image bounds (warning if outside)
- Category names are non-empty strings
- Image files exist (optional, use --skip-image-check to disable)
- Reject legacy geometry keys: bbox, polygon, line, line_points

Examples:

  # Full validation
  python validate_jsonl.py lvis/processed/train.jsonl
  
  # Skip image existence check (faster)
  python validate_jsonl.py lvis/processed/train.jsonl --skip-image-check
  
  # Verbose output
  python validate_jsonl.py lvis/processed/train.jsonl --verbose
        """
    )
    
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="JSONL file to validate"
    )
    
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip checking if image files exist (faster)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    validator = JSONLValidator(
        check_images=not args.skip_image_check,
        verbose=args.verbose
    )
    
    success = validator.validate_file(args.jsonl_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
