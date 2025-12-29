import os
import json
import traceback
from pathlib import Path
import cv2
import numpy as np
import pyvips
import openslide
from opensdpc import OpenSdpc


# --- Helper Functions ---

def get_slide_handler(path):
    """Abstracts slide opening for SDPC and OpenSlide."""
    ext = Path(path).suffix.lower()
    if ext == '.sdpc':
        slide = OpenSdpc(str(path))
        dims = slide.level_dimensions[0]
    else:
        slide = openslide.OpenSlide(str(path))
        dims = slide.dimensions
            
    return slide, dims


def load_json_bbox(json_path):
    """Loads bounding box from your custom JSON format."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return None

    # Navigate to coordinates
    try:
        labels = data.get('GroupModel', {}).get('Labels', [])
        if not labels:
            print("❌ No Labels found in JSON")
            return None

        # Get first label's coordinates
        coords = labels[0].get('Coordinates', [])
        if len(coords) < 2:
            print("❌ Invalid coordinates in JSON")
            return None

        # Extract bounding box from two corner points
        x1, y1 = coords[0]['X'], coords[0]['Y']
        x2, y2 = coords[1]['X'], coords[1]['Y']

        # Ensure min/max order
        minx, maxx = min(x1, x2), max(x1, x2)
        miny, maxy = min(y1, y2), max(y1, y2)

        width = maxx - minx
        height = maxy - miny
        area_pixels = width * height

        print(f"  📦 JSON ROI coordinates:")
        print(f"     Top-left: ({minx:.0f}, {miny:.0f})")
        print(f"     Bottom-right: ({maxx:.0f}, {maxy:.0f})")
        print(f"  📐 ROI Size: {width:.0f} x {height:.0f} pixels")
        print(f"  📊 ROI Area: {area_pixels / 1e6:.2f} megapixels")

        return (minx, miny, maxx, maxy)

    except Exception as e:
        print(f"❌ Error parsing JSON structure: {e}")
        traceback.print_exc()
        return None


def save_visualization(slide, bbox, save_path):
    """Saves a thumbnail with drawn bounding box."""
    try:
        # Check if slide has get_thumbnail method with integer argument (SDPC)
        if hasattr(slide, 'level_downsamples'):
            thumb_raw = slide.get_thumbnail(3)  # Level 3
            
            # Convert PIL Image to numpy if needed
            if hasattr(thumb_raw, 'mode'):  # PIL Image
                thumb = np.array(thumb_raw, dtype=np.uint8)
            elif isinstance(thumb_raw, np.ndarray):
                thumb = thumb_raw
            else:
                thumb = np.array(thumb_raw, dtype=np.uint8)
            
            scale = slide.level_downsamples[3]
        else:
            # OpenSlide format
            thumb_raw = slide.get_thumbnail((2048, 2048))
            
            # Convert PIL Image to numpy if needed
            if hasattr(thumb_raw, 'mode'):  # PIL Image
                thumb = np.array(thumb_raw, dtype=np.uint8)
            elif isinstance(thumb_raw, np.ndarray):
                thumb = thumb_raw
            else:
                thumb = np.array(thumb_raw, dtype=np.uint8)
            
            scale = slide.dimensions[0] / thumb.shape[1]

        # Validate the numpy array
        
        if not isinstance(thumb, np.ndarray) or thumb.size == 0:
            print(f"  ⚠️  Invalid thumbnail data: not a valid numpy array")
            return
        
        if len(thumb.shape) < 2:
            print(f"  ⚠️  Invalid thumbnail shape: {thumb.shape}")
            return
        
        # Force create a real numpy array copy to avoid any memory view issues
        # This is critical for OpenCV compatibility
        thumb = np.array(thumb, dtype=np.uint8, copy=True)

        # Resize if thumbnail is still too large (max dimension 2048)
        max_dim = 2048
        h, w = thumb.shape[:2]
        if max(h, w) > max_dim:
            resize_factor = max_dim / max(h, w)
            new_w = int(w * resize_factor)
            new_h = int(h * resize_factor)
            
            thumb = cv2.resize(thumb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale = scale / resize_factor

        vis = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)

        # Draw bounding box
        minx, miny, maxx, maxy = bbox
        pt1 = (int(minx / scale), int(miny / scale))
        pt2 = (int(maxx / scale), int(maxy / scale))
        cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 4)

        cv2.imwrite(str(save_path), vis)
        print(f"  🖼️  Visualization saved: {Path(save_path).name}")
    except Exception as e:
        print(f"  ⚠️  Visualization failed: {e}")
        traceback.print_exc()


def save_pyramid_tiff(img_np, save_path, target_mpp, scale_factor=1.0, 
                      quality=75, bg_threshold=240, tile_size=256,
                      compression='jpeg'):
    """Saves numpy array as pyramid TIFF using pyvips with JPEG compression.
    
    Optimized for minimal file size with good visual quality.
    """
    h, w, c = img_np.shape

    print(f"\n  💾 Saving pyramid TIFF...")
    print(f"     Input: {w} x {h} x {c}")
    print(f"     Input memory: {img_np.nbytes / 1024 / 1024:.1f} MB")

    try:
        # Ensure contiguous array before creating vips image
        if not img_np.flags['C_CONTIGUOUS']:
            img_np = np.ascontiguousarray(img_np)

        # Create vips image from numpy
        print(f"     Creating vips image...")
        vips_img = pyvips.Image.new_from_memory(img_np.data, w, h, c, 'uchar')
        if c == 3:
            vips_img = vips_img.copy(interpretation='srgb')
        
        print(f"     ✓ Vips image created: {vips_img.width} x {vips_img.height}")
        
        # Free numpy array memory early (vips has its own copy now)
        del img_np

        # Resize if needed (only if scale difference is significant)
        if abs(scale_factor - 1.0) > 0.01:
            print(f"     Resizing with scale factor {scale_factor:.3f}...")
            vips_img = vips_img.resize(scale_factor, kernel='lanczos3')
            print(f"     ✓ Resized to: {vips_img.width} x {vips_img.height}")
            if vips_img.format != 'uchar':
                vips_img = vips_img.cast('uchar')
        
        # Background cleaning AFTER resize (faster on smaller image)
        # Note: For JPEG, white areas compress well anyway, so this is optional
        if bg_threshold is not None:
            print(f"     🧹 Cleaning background (Threshold > {bg_threshold})...")
            # Convert to numpy only for the cleaning operation
            img_array = vips_img.numpy()
            if c == 3:
                mask = np.min(img_array, axis=2) > bg_threshold
                img_array[mask] = 255
            else:
                img_array[img_array > bg_threshold] = 255
            
            # Recreate vips image from cleaned array
            vips_img = pyvips.Image.new_from_memory(
                img_array.data, 
                img_array.shape[1], 
                img_array.shape[0], 
                c, 
                'uchar'
            )
            if c == 3:
                vips_img = vips_img.copy(interpretation='srgb')
            del img_array  # Free memory
            print(f"     ✓ Background cleaned")

        # Optimized JPEG compression parameters for minimal file size
        print(f"     Compression: JPEG Q={quality}, Pyramid, Tiled {tile_size}x{tile_size}")
        print(f"     Writing to disk (this may take a while)...")
        
        vips_img.write_to_file(
            str(save_path),
            compression='jpeg',
            Q=quality,
            tile=True,
            tile_width=tile_size,
            tile_height=tile_size,
            pyramid=True,
            subifd=False,           # QuPath compatibility
            bigtiff=True,
            xres=10000.0 / target_mpp,
            yres=10000.0 / target_mpp,
            resunit='cm',
        )

        # Check output file
        if Path(save_path).exists():
            output_size_mb = Path(save_path).stat().st_size / 1024 / 1024
            # Calculate compression ratio based on final vips image dimensions
            original_size_mb = (vips_img.width * vips_img.height * c) / 1024 / 1024
            compression_ratio = original_size_mb / output_size_mb
            
            print(f"  ✅ Saved successfully!")
            print(f"     Output file: {output_size_mb:.1f} MB")
            print(f"     Compression ratio: {compression_ratio:.1f}x")
            print(f"     Path: {save_path}")
            return True
        else:
            print(f"  ❌ File was not created!")
            return False

    except Exception as e:
        print(f"  ❌ Save failed: {e}")
        traceback.print_exc()
        return False


def extract_and_save_roi_from_json(wsi_path, json_path, save_path,
                                   padding=1000,
                                   target_mpp=0.25, manual_mpp=None,
                                   visualize=False, 
                                   quality=75,  # 降低默认值
                                   bg_threshold=240, 
                                   tile_size=256,  # 降低默认值
                                   compression='jpeg',  # 新增参数
                                   **kwargs):
    """Extract ROI from WSI using JSON coordinate file."""
    wsi_path, save_path = Path(wsi_path), Path(save_path)
    print(f"\n🔹 Processing: {wsi_path.name}")

    try:
        slide, dims = get_slide_handler(wsi_path)
        src_mpp = manual_mpp if manual_mpp else 0.25

        print(f"  🔬 Slide info: {dims[0]} x {dims[1]} pixels")
        print(f"  📏 Source MPP: {src_mpp:.3f}, Target MPP: {target_mpp:.3f}")

        # Load ROI bounding box from JSON
        bbox = load_json_bbox(json_path)
        if bbox is None:
            print("  ❌ No valid ROI found in JSON.")
            return False

        if visualize:
            save_visualization(slide, bbox, str(save_path).replace('.tiff', '_vis.png'))

        # Calculate padded region
        minx, miny, maxx, maxy = bbox

        # ROI dimensions (without padding)
        roi_w = int(maxx - minx)
        roi_h = int(maxy - miny)

        print(f"\n  🎯 Extracting region:")
        print(f"     Original ROI: {roi_w} x {roi_h} pixels")

        # Apply padding and clamp to slide boundaries
        x = max(0, int(minx) - padding)
        y = max(0, int(miny) - padding)
        w = min(int(maxx) + padding, dims[0]) - x
        h = min(int(maxy) + padding, dims[1]) - y

        print(f"     With padding ({padding}px): {w} x {h} pixels")
        print(f"     Position: x={x}, y={y}")

        # Sanity checks
        if w <= 0 or h <= 0:
            print(f"  ❌ Invalid region dimensions: {w} x {h}")
            return False

        if w > dims[0] or h > dims[1]:
            print(f"  ⚠️  Warning: Region larger than slide!")

        # Check for extremely large regions
        estimated_mem = w * h * 3 / 1024 / 1024 / 1024  # GB
        if estimated_mem > 16:  # Warning if > 16GB
            print(f"  ⚠️  WARNING: Region is extremely large ({estimated_mem:.1f} GB raw). This might cause memory issues.")

        # Read region
        print(f"  📖 Reading region from slide...")
        region = slide.read_region((x, y), 0, (w, h))
        
        # Convert to numpy and remove alpha channel in one step
        if hasattr(region, 'mode') and region.mode == 'RGBA':
            # PIL Image with alpha - convert to RGB directly
            region = region.convert('RGB')
            img_np = np.array(region, dtype=np.uint8)
        else:
            img_np = np.array(region, dtype=np.uint8)
            if img_np.shape[2] == 4:
                img_np = img_np[..., :3]
        
        print(f"  ✅ Extracted image: {img_np.shape[1]} x {img_np.shape[0]} x {img_np.shape[2]}")
        print(f"  💾 Raw size: {img_np.nbytes / 1024 / 1024:.1f} MB")

        # Calculate scale factor
        scale_factor = src_mpp / target_mpp
        if abs(scale_factor - 1.0) > 0.01:
            new_w = int(img_np.shape[1] * scale_factor)
            new_h = int(img_np.shape[0] * scale_factor)
            print(f"  🔄 Will resample to: {new_w} x {new_h} (x{scale_factor:.2f})")

        return save_pyramid_tiff(img_np, save_path, target_mpp, scale_factor, 
                                quality=quality, bg_threshold=bg_threshold, 
                                tile_size=tile_size, compression=compression)

    except Exception as e:
        print(f"  ❌ Error: {e}")
        traceback.print_exc()
        return False


def batch_process_folder(input_dir, output_dir, file_pattern="*.sdpc", **kwargs):
    """Batch process WSI files and their JSON annotations from the same folder."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Only search in the specified folder, no recursion
    files = [f for f in input_dir.glob(file_pattern) if f.is_file()]
    print(f"🚀 Batch Processing {len(files)} files from {input_dir}")

    stats = {'ok': 0, 'fail': 0, 'skip': 0}

    for i, wsi_file in enumerate(files, 1):
        json_file = wsi_file.with_suffix('.json')
        out_path = output_dir / (wsi_file.stem + ".tiff")

        if not json_file.exists():
            print(f"\n[{i}/{len(files)}] ⚠️  Skipping {wsi_file.name} (No matching JSON)");
            stats['skip'] += 1
            continue
        if out_path.exists():
            print(f"\n[{i}/{len(files)}] ⏭️  Skipping {wsi_file.name} (Already exists)");
            stats['skip'] += 1
            continue

        print(f"\n[{i}/{len(files)}] Processing {wsi_file.name}...")
        if extract_and_save_roi_from_json(wsi_file, json_file, out_path, **kwargs):
            stats['ok'] += 1
        else:
            stats['fail'] += 1

    print(f"\n🏁 Done. Success: {stats['ok']}, Failed: {stats['fail']}, Skipped: {stats['skip']}")


if __name__ == "__main__":
    # # 单文件处理示例
    # extract_and_save_roi_from_json(
    #     wsi_path="/mnt/6T/GML/DATA/WSI/Reactive-Hyperplasia/B2023-36262/B2023-36262-D.sdpc",
    #     json_path="/mnt/6T/GML/DATA/WSI/Reactive-Hyperplasia/B2023-36262/B2023-36262-D.json",
    #     save_path="/mnt/6T/GML/DATA/WSI/Reactive-Hyperplasia/B2023-36262/B2023-36262-D.tiff",
    #     target_mpp=0.104,
    #     manual_mpp=0.104,
    #     padding=2000,
    #     visualize=True
    # )

    extract_and_save_roi_from_json(
        wsi_path="/mnt/6T/GML/DATA/WSI/MALT-Lymphoma/B2018-06208/ToRegister/B2018-06208B-cd3.sdpc",
        json_path="/mnt/6T/GML/DATA/WSI/MALT-Lymphoma/B2018-06208/ToRegister/B2018-06208B-cd3.json",
        save_path="/mnt/6T/GML/DATA/WSI/MALT-Lymphoma/B2018-06208/ToRegister/B2018-06208B-cd3.tiff",
        target_mpp=0.104074,
        manual_mpp=0.104074,
        padding=0,
        visualize=True,
        compression='jpeg',
        quality=85,            
        tile_size=512,        
        bg_threshold=240
    )   


    # 批量处理示例 - SDPC和JSON在同一文件夹
    # batch_process_folder(
    #     input_dir="/mnt/6T/GML/DATA/WSI/MALT-Lymphoma/xsB2022-13516",
    #     output_dir="/mnt/6T/GML/DATA/WSI/MALT-Lymphoma/xsB2022-13516",
    #     file_pattern="*.sdpc",
    #     target_mpp=0.104,
    #     manual_mpp=0.104,
    #     padding=2000,
    #     visualize=True,
    #     quality=80,        # 恢复为高质量
    #     bg_threshold=240,  # 开启背景清洗
    # )