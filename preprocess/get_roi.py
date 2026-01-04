import os
import json
import traceback
from pathlib import Path
import cv2
import numpy as np
import pyvips
import openslide
from opensdpc import OpenSdpc

# --- Libvips Performance Tuning ---
# Control concurrency (set to number of cores you want to use, or 0 for auto)
# pyvips.cache_set_max(0)  # Uncomment to disable cache (reduces memory)
# pyvips.concurrency_set(4)  # Uncomment to limit threads (reduces CPU load)


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
        # Check if slide is OpenSlide (standard) or SDPC (custom)
        # OpenSlide objects also have 'level_downsamples', so we must check type explicitly
        if not isinstance(slide, openslide.OpenSlide):
            # Assume SDPC or other custom format that takes int level
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
            # OpenSlide format - takes size tuple
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


def save_pyramid_tiff(img_data, save_path, target_mpp, scale_factor=1.0, 
                      quality=75, bg_threshold=240, tile_size=256,
                      compression='jpeg'):
    """Saves image data as pyramid TIFF using pyvips.
    
    Args:
        img_data: numpy array or pyvips.Image
    """
    print(f"\n  💾 Saving pyramid TIFF...")
    
    try:
        # 1. Create vips image from input
        if isinstance(img_data, np.ndarray):
            h, w, c = img_data.shape
            print(f"     Input (Numpy): {w} x {h} x {c}")
            print(f"     Input memory: {img_data.nbytes / 1024 / 1024:.1f} MB")
            
            if not img_data.flags['C_CONTIGUOUS']:
                img_data = np.ascontiguousarray(img_data)
                
            vips_img = pyvips.Image.new_from_memory(img_data.data, w, h, c, 'uchar')
            if c == 3:
                vips_img = vips_img.copy(interpretation='srgb')
            del img_data # Free numpy memory
            
        elif isinstance(img_data, pyvips.Image):
            vips_img = img_data
            c = vips_img.bands
            print(f"     Input (PyVips): {vips_img.width} x {vips_img.height} x {c}")
        else:
            raise ValueError(f"Unsupported image type: {type(img_data)}")

        print(f"     ✓ Vips image ready: {vips_img.width} x {vips_img.height}")

        # 2. Resize
        if abs(scale_factor - 1.0) > 0.01:
            print(f"     Resizing with scale factor {scale_factor:.3f}...")
            vips_img = vips_img.resize(scale_factor, kernel='lanczos3')
            print(f"     ✓ Resized to: {vips_img.width} x {vips_img.height}")
            if vips_img.format != 'uchar':
                vips_img = vips_img.cast('uchar')
        
        # 3. Background cleaning (Pure Vips implementation)
        # Note: This step adds computational overhead. Set bg_threshold=None to disable.
        if bg_threshold is not None:
            print(f"     🧹 Cleaning background (Threshold > {bg_threshold})...")
            # Create mask: if all channels > threshold, it's background
            if vips_img.bands >= 3:
                # Universal implementation using ifthenelse (works on all libvips versions)
                bands = vips_img.bandsplit()
                min_val = bands[0]
                for band in bands[1:]:
                    # Pixel-wise min: if min_val < band, keep min_val, else use band
                    min_val = (min_val < band).ifthenelse(min_val, band)
                mask = min_val > bg_threshold
            else:
                mask = vips_img > bg_threshold
            
            # Apply mask: if background, set to 255, else keep original
            vips_img = mask.ifthenelse(255, vips_img)
            print(f"     ✓ Background cleaned")

        # 4. Save
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
            subifd=False,
            bigtiff=True,
            xres=1000.0 / target_mpp,
            yres=1000.0 / target_mpp,
            resunit='cm',
        )

        # Check output
        if Path(save_path).exists():
            output_size_mb = Path(save_path).stat().st_size / 1024 / 1024
            original_size_mb = (vips_img.width * vips_img.height * c) / 1024 / 1024
            compression_ratio = original_size_mb / output_size_mb if output_size_mb > 0 else 0
            
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


def process_openslide_optimized(wsi_path, bbox, save_path, padding, target_mpp, manual_mpp, 
                                visualize, quality, bg_threshold, tile_size, compression):
    """Optimized processing for OpenSlide supported formats using pure libvips pipeline."""
    print(f"  🚀 Using optimized libvips pipeline for OpenSlide...")
    
    try:
        # 1. Open image directly with pyvips (lazy loading)
        # access='sequential' is usually faster for single pass, but 'random' is safer for ROI extraction
        # Let's try default (random) first as we are extracting a specific region
        vips_src = pyvips.Image.new_from_file(str(wsi_path))
        
        # Get dimensions
        dims = (vips_src.width, vips_src.height)
        src_mpp = manual_mpp if manual_mpp else 0.25 # TODO: Try to read from vips metadata if possible
        
        print(f"  🔬 Slide info: {dims[0]} x {dims[1]} pixels")
        print(f"  📏 Source MPP: {src_mpp:.3f}, Target MPP: {target_mpp:.3f}")
        
        # 2. Visualization (Optional - requires separate OpenSlide handle or vips thumbnail)
        if visualize:
            try:
                # Use a temporary OpenSlide handle just for the thumbnail to keep logic simple
                # Or use vips thumbnail
                thumb = vips_src.thumbnail_image(2048)
                thumb_np = np.ndarray(buffer=thumb.write_to_memory(),
                                      dtype=np.uint8,
                                      shape=[thumb.height, thumb.width, thumb.bands])
                
                # Calculate scale for drawing bbox
                vis_scale = dims[0] / thumb.width
                
                vis = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2BGR)
                minx, miny, maxx, maxy = bbox
                pt1 = (int(minx / vis_scale), int(miny / vis_scale))
                pt2 = (int(maxx / vis_scale), int(maxy / vis_scale))
                cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 4)
                
                vis_path = str(save_path).replace('.tiff', '_vis.png')
                cv2.imwrite(vis_path, vis)
                print(f"  🖼️  Visualization saved: {Path(vis_path).name}")
            except Exception as e:
                print(f"  ⚠️  Visualization failed: {e}")

        # 3. Calculate ROI
        minx, miny, maxx, maxy = bbox
        x = max(0, int(minx) - padding)
        y = max(0, int(miny) - padding)
        w = min(int(maxx) + padding, dims[0]) - x
        h = min(int(maxy) + padding, dims[1]) - y
        
        print(f"  🎯 Extracting region (Lazy):")
        print(f"     With padding ({padding}px): {w} x {h} pixels")
        print(f"     Position: x={x}, y={y}")
        
        # 4. Extract Area (Lazy)
        roi = vips_src.extract_area(x, y, w, h)
        
        # Remove alpha if present
        if roi.bands == 4:
            roi = roi.extract_band(0, n=3)
            
        # 5. Calculate scale factor
        scale_factor = src_mpp / target_mpp
        if abs(scale_factor - 1.0) > 0.01:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            print(f"  🔄 Will resample to: {new_w} x {new_h} (x{scale_factor:.2f})")
            
        # 6. Save (Triggers the pipeline)
        return save_pyramid_tiff(roi, save_path, target_mpp, scale_factor, 
                                quality=quality, bg_threshold=bg_threshold, 
                                tile_size=tile_size, compression=compression)
                                
    except Exception as e:
        print(f"  ❌ Optimized processing failed: {e}")
        traceback.print_exc()
        return False


def extract_and_save_roi_from_json(wsi_path, json_path, save_path,
                                   padding=1000,
                                   target_mpp=0.25, manual_mpp=None,
                                   visualize=False, 
                                   quality=75,  # 降低默认值
                                   bg_threshold=None, 
                                   tile_size=256,  # 降低默认值
                                   compression='jpeg',  # 新增参数
                                   **kwargs):
    """Extract ROI from WSI using JSON coordinate file."""
    wsi_path, save_path = Path(wsi_path), Path(save_path)
    print(f"\n🔹 Processing: {wsi_path.name}")

    # Load ROI bounding box from JSON first
    bbox = load_json_bbox(json_path)
    if bbox is None:
        print("  ❌ No valid ROI found in JSON.")
        return False

    # Check if we can use optimized path (OpenSlide supported files)
    # SDPC files must use the old path
    is_sdpc = wsi_path.suffix.lower() == '.sdpc'
    
    if not is_sdpc:
        return process_openslide_optimized(wsi_path, bbox, save_path, padding, target_mpp, 
                                          manual_mpp, visualize, quality, bg_threshold, 
                                          tile_size, compression)

    # --- Fallback / SDPC Legacy Path ---
    try:
        slide, dims = get_slide_handler(wsi_path)
        src_mpp = manual_mpp if manual_mpp else 0.25

        print(f"  🔬 Slide info: {dims[0]} x {dims[1]} pixels")
        print(f"  📏 Source MPP: {src_mpp:.3f}, Target MPP: {target_mpp:.3f}")

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
        wsi_path="/home/william/Downloads/B2018-06208B-cd3.svs",
        json_path="/home/william/Downloads/B2018-06208B-cd3.json",
        save_path="/home/william/Downloads/B2018-06208B-cd3.tiff",
        target_mpp=0.104074,
        manual_mpp=0.104074,
        padding=1000,
        visualize=True,
        compression='jpeg',
        quality=80,
        tile_size=512,        
        bg_threshold=None
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