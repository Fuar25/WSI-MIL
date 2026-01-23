import os
import json
import re
import traceback
import logging
import time
import gc
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pyvips
import openslide
from opensdpc import OpenSdpc

MAX_MEMORY_GB = 36  # Maximum allowed memory for ROI extraction in GB
# --- Libvips Performance Tuning ---
# Control concurrency (set to number of cores you want to use, or 0 for auto)
# pyvips.cache_set_max(0)  # Uncomment to disable cache (reduces memory)
# pyvips.concurrency_set(4)  # Uncomment to limit threads (reduces CPU load)


# --- Helper Functions ---

def setup_logging(output_dir, log_name="roi_extraction.log"):
    """Setup logging configuration with both file and console handlers."""
    log_path = Path(output_dir) / log_name
    
    # Create logger
    logger = logging.getLogger('roi_extraction')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler - detailed logs
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    
    # Console handler - important info only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
    
    return logger


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
    """Loads bounding boxes from your custom JSON format.
    Returns a list of tuples: [(minx, miny, maxx, maxy), ...]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return []

    # Navigate to coordinates
    try:
        labels = data.get('GroupModel', {}).get('Labels', [])
        if not labels:
            print("❌ No Labels found in JSON")
            return []

        bboxes = []
        for label in labels:
            # Get label's coordinates
            coords = label.get('Coordinates', [])
            if len(coords) < 2:
                print(f"⚠️ Invalid coordinates in label ID {label.get('ID')}")
                continue

            # Extract bounding box from two corner points
            x1, y1 = coords[0]['X'], coords[0]['Y']
            x2, y2 = coords[1]['X'], coords[1]['Y']

            # Ensure min/max order
            minx, maxx = min(x1, x2), max(x1, x2)
            miny, maxy = min(y1, y2), max(y1, y2)

            width = maxx - minx
            height = maxy - miny
            area_pixels = width * height

            print(f"  📦 JSON ROI coordinates (ID: {label.get('ID')}):")
            print(f"     Top-left: ({minx:.0f}, {miny:.0f})")
            print(f"     Bottom-right: ({maxx:.0f}, {maxy:.0f})")
            print(f"  📐 ROI Size: {width:.0f} x {height:.0f} pixels")
            print(f"  📊 ROI Area: {area_pixels / 1e6:.2f} megapixels")
            
            bboxes.append((minx, miny, maxx, maxy))

        return bboxes

    except Exception as e:
        print(f"❌ Error parsing JSON structure: {e}")
        traceback.print_exc()
        return []


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

    # Load ROI bounding boxes from JSON first
    bboxes = load_json_bbox(json_path)
    if not bboxes:
        print("  ❌ No valid ROI found in JSON.")
        return False

    # Check if we can use optimized path (OpenSlide supported files)
    is_sdpc = wsi_path.suffix.lower() == '.sdpc'
    
    # Helper function to process a single bbox
    def process_single_roi(bbox, current_save_path):
        if not is_sdpc:
            return process_openslide_optimized(wsi_path, bbox, current_save_path, padding, target_mpp, 
                                              manual_mpp, visualize, quality, bg_threshold, 
                                              tile_size, compression)
        else:
            # --- Fallback / SDPC Legacy Path ---
            slide = None
            img_np = None
            try:
                slide, dims = get_slide_handler(wsi_path)
                src_mpp = manual_mpp if manual_mpp else 0.25

                print(f"  🔬 Slide info: {dims[0]} x {dims[1]} pixels")
                print(f"  📏 Source MPP: {src_mpp:.3f}, Target MPP: {target_mpp:.3f}")

                if visualize:
                    save_visualization(slide, bbox, str(current_save_path).replace('.tiff', '_vis.png'))

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
                if estimated_mem > MAX_MEMORY_GB:  # Limit to 12GB to prevent OOM
                    print(f"  ❌ ERROR: Region is too large ({estimated_mem:.1f} GB raw). Skipping to prevent OOM.")
                    return False

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
                
                # Free PIL image memory immediately
                del region

                print(f"  ✅ Extracted image: {img_np.shape[1]} x {img_np.shape[0]} x {img_np.shape[2]}")
                print(f"  💾 Raw size: {img_np.nbytes / 1024 / 1024:.1f} MB")

                # Calculate scale factor
                scale_factor = src_mpp / target_mpp
                if abs(scale_factor - 1.0) > 0.01:
                    new_w = int(img_np.shape[1] * scale_factor)
                    new_h = int(img_np.shape[0] * scale_factor)
                    print(f"  🔄 Will resample to: {new_w} x {new_h} (x{scale_factor:.2f})")

                result = save_pyramid_tiff(img_np, current_save_path, target_mpp, scale_factor, 
                                        quality=quality, bg_threshold=bg_threshold, 
                                        tile_size=tile_size, compression=compression)
                
                # Free numpy array memory
                del img_np
                return result

            except Exception as e:
                print(f"  ❌ Error: {e}")
                traceback.print_exc()
                return False
            finally:
                # Ensure slide is closed
                if slide is not None and hasattr(slide, 'close'):
                    try:
                        slide.close()
                    except:
                        pass
                # Force garbage collection
                gc.collect()

    # Logic for handling multiple ROIs
    if len(bboxes) == 1:
        return process_single_roi(bboxes[0], save_path)
    
    elif len(bboxes) == 2:
        print(f"  🔢 Found 2 ROIs. Attempting to split based on filename...")
        # Sort by X coordinate (minx)
        bboxes.sort(key=lambda b: b[0])
        
        # Parse filename
        filename = wsi_path.stem
        # Regex to match PathologyNumber + TissueCodes (2 letters) + Suffix
        # e.g. B2022-25838AB-cd20 -> B2022-25838, AB, -cd20
        match = re.search(r'([a-zA-Z0-9]+-\d+)([A-Z]{2})(.*)', filename)
        
        if match:
            pathology_num = match.group(1)
            tissue_codes = match.group(2)
            suffix = match.group(3)
            
            print(f"     Pathology Number: {pathology_num}")
            print(f"     Tissue Codes: {tissue_codes}")
            print(f"     Suffix: {suffix}")
            
            results = []
            
            # Left ROI -> First letter
            left_name = f"{pathology_num}{tissue_codes[0]}{suffix}"
            left_path = save_path.with_name(left_name + save_path.suffix)
            print(f"     ⬅️ Left ROI -> {left_path.name}")
            results.append(process_single_roi(bboxes[0], left_path))
            
            # Right ROI -> Second letter
            right_name = f"{pathology_num}{tissue_codes[1]}{suffix}"
            right_path = save_path.with_name(right_name + save_path.suffix)
            print(f"     ➡️ Right ROI -> {right_path.name}")
            results.append(process_single_roi(bboxes[1], right_path))
            
            return all(results)
        else:
            print(f"  ⚠️  Filename '{filename}' does not match expected pattern for splitting (e.g., '...AB...').")
            print(f"  ⚠️  Processing only the first ROI as fallback.")
            return process_single_roi(bboxes[0], save_path)
            
    else:
        print(f"  ⚠️  Found {len(bboxes)} ROIs. Only 1 or 2 are supported for automatic splitting.")
        print(f"  ⚠️  Processing only the first ROI.")
        return process_single_roi(bboxes[0], save_path)


def batch_process_folder(input_dir, output_dir, 
                        file_pattern="*.sdpc",
                        recursive=True,
                        padding=1000,
                        target_mpp=0.25, 
                        manual_mpp=None,
                        visualize=False,
                        quality=75,
                        bg_threshold=None,
                        tile_size=256,
                        compression='jpeg'):
    """Batch process WSI files and their JSON annotations.
    
    Args:
        input_dir: Root directory to search for WSI files
        output_dir: Directory to save extracted ROIs
        file_pattern: File pattern to match (e.g., "*.sdpc", "*.svs")
        recursive: If True, recursively search subdirectories
        padding: Padding around ROI in pixels
        target_mpp: Target microns per pixel for output
        manual_mpp: Manual override for source MPP (None = auto-detect)
        visualize: Save thumbnail with ROI overlay
        quality: JPEG quality (1-100)
        bg_threshold: Background threshold for cleaning (None = disable)
        tile_size: Tile size for pyramid TIFF
        compression: Compression method ('jpeg', 'deflate', etc.)
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Log session start
    logger.info("="*80)
    logger.info(f"🚀 Batch ROI Extraction Started")
    logger.info(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Input Directory: {input_dir}")
    logger.info(f"   Output Directory: {output_dir}")
    logger.info(f"   File Pattern: {file_pattern}")
    logger.info(f"   Recursive Search: {recursive}")
    logger.info(f"   Parameters: padding={padding}, target_mpp={target_mpp}, ")
    logger.info(f"               manual_mpp={manual_mpp}, quality={quality}, ")
    logger.info(f"               tile_size={tile_size}, compression={compression}, ")
    logger.info(f"               bg_threshold={bg_threshold}, visualize={visualize}")
    logger.info("="*80)
    
    # Search for files (recursive or not)
    search_func = input_dir.rglob if recursive else input_dir.glob
    files = [f for f in search_func(file_pattern) if f.is_file()]
    logger.info(f"\n📂 Found {len(files)} files matching pattern '{file_pattern}'")
    
    if not files:
        logger.warning("⚠️  No files found. Exiting.")
        return
    
    # Statistics
    stats = {'ok': 0, 'fail': 0, 'skip_no_json': 0, 'skip_exists': 0}
    start_time = time.time()
    
    for i, wsi_file in enumerate(files, 1):
        file_start_time = time.time()
        
        # Find matching JSON file (same name, same directory)
        json_file = wsi_file.with_suffix('.json')
        
        # Determine output path (preserve relative structure if recursive)
        if recursive:
            rel_path = wsi_file.relative_to(input_dir)
            out_path = output_dir / rel_path.parent / (wsi_file.stem + ".tiff")
            out_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            out_path = output_dir / (wsi_file.stem + ".tiff")
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"[{i}/{len(files)}] Processing: {wsi_file.name}")
        logger.info(f"   Source: {wsi_file}")
        logger.info(f"   Output: {out_path}")
        
        # Check for JSON
        if not json_file.exists():
            logger.warning(f"   ⚠️  SKIPPED: No matching JSON file found")
            logger.warning(f"   Expected: {json_file}")
            stats['skip_no_json'] += 1
            continue
        
        # Check if output already exists
        if out_path.exists():
            existing_size = out_path.stat().st_size / 1024 / 1024
            logger.info(f"   ⏭️  SKIPPED: Output file already exists ({existing_size:.1f} MB)")
            stats['skip_exists'] += 1
            continue
            
        # Check for split outputs (for 2 ROIs case)
        # Regex to match PathologyNumber + TissueCodes (2 letters) + Suffix
        match = re.search(r'([a-zA-Z0-9]+-\d+)([A-Z]{2})(.*)', wsi_file.stem)
        if match:
            pathology_num = match.group(1)
            tissue_codes = match.group(2)
            suffix_part = match.group(3)
            
            # Construct expected split filenames (Left and Right)
            # Must match logic in extract_and_save_roi_from_json
            ext = out_path.suffix
            
            left_name = f"{pathology_num}{tissue_codes[0]}{suffix_part}{ext}"
            right_name = f"{pathology_num}{tissue_codes[1]}{suffix_part}{ext}"
            
            left_path = out_path.with_name(left_name)
            right_path = out_path.with_name(right_name)
            
            if left_path.exists() and right_path.exists():
                logger.info(f"   ⏭️  SKIPPED: Split output files already exist:")
                logger.info(f"       - {left_path.name}")
                logger.info(f"       - {right_path.name}")
                stats['skip_exists'] += 1
                continue
        
        # Get input file size
        input_size = wsi_file.stat().st_size / 1024 / 1024
        logger.info(f"   📊 Input Size: {input_size:.1f} MB")
        
        # Process the slide
        try:
            success = extract_and_save_roi_from_json(
                wsi_path=wsi_file,
                json_path=json_file,
                save_path=out_path,
                padding=padding,
                target_mpp=target_mpp,
                manual_mpp=manual_mpp,
                visualize=visualize,
                quality=quality,
                bg_threshold=bg_threshold,
                tile_size=tile_size,
                compression=compression
            )
            
            processing_time = time.time() - file_start_time
            
            if success:
                if out_path.exists():
                    output_size = out_path.stat().st_size / 1024 / 1024
                    logger.info(f"   ✅ SUCCESS")
                    logger.info(f"   📦 Output Size: {output_size:.1f} MB")
                    logger.info(f"   ⏱️  Processing Time: {processing_time:.1f} seconds")
                    logger.info(f"   💾 Saved to: {out_path}")
                else:
                    logger.info(f"   ✅ SUCCESS (Split into multiple files)")
                    logger.info(f"   ⏱️  Processing Time: {processing_time:.1f} seconds")
                    logger.info(f"   💾 Saved to: {out_path.parent} (See split files)")
                stats['ok'] += 1
            else:
                logger.error(f"   ❌ FAILED: Processing returned False")
                logger.error(f"   ⏱️  Time spent: {processing_time:.1f} seconds")
                stats['fail'] += 1
                
        except Exception as e:
            processing_time = time.time() - file_start_time
            logger.error(f"   ❌ FAILED: {str(e)}")
            logger.error(f"   ⏱️  Time spent: {processing_time:.1f} seconds")
            logger.debug(f"   Stack trace:", exc_info=True)
            stats['fail'] += 1
        
        # Force garbage collection after each file
        gc.collect()
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"🏁 Batch Processing Complete")
    logger.info(f"   Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"   ✅ Successful: {stats['ok']}")
    logger.info(f"   ❌ Failed: {stats['fail']}")
    logger.info(f"   ⏭️  Skipped (already exists): {stats['skip_exists']}")
    logger.info(f"   ⚠️  Skipped (no JSON): {stats['skip_no_json']}")
    logger.info(f"   📊 Total Files: {len(files)}")
    if stats['ok'] > 0:
        logger.info(f"   ⏱️  Average Time: {total_time/stats['ok']:.1f} seconds per file")
    logger.info("="*80)


if __name__ == "__main__":
    # extract_and_save_roi_from_json(
    #     wsi_path="/mnt/6T/GML/DATA/WSI/SDPC/MALT/Ki-67/hasROI/B2019-36238AB-ki67.sdpc",
    #     json_path="/mnt/6T/GML/DATA/WSI/SDPC/MALT/Ki-67/hasROI/B2019-36238AB-ki67.json",
    #     save_path="/mnt/6T/GML/DATA/WSI/SDPC/MALT/Ki-67/hasROI/B2019-36238AB-ki67.tiff",
    #     target_mpp=0.104074,
    #     manual_mpp=0.104074,
    #     padding=1000,
    #     visualize=True,
    #     compression='jpeg',
    #     quality=85,
    #     tile_size=512,        
    #     bg_threshold=None
    # )   


    # 批量处理示例 - 递归搜索所有子文件夹
    batch_process_folder(
        input_dir="/mnt/6T/GML/DATA/WSI/Others/Reactive-Hyperplasia/xsB2023-01220",
        output_dir="/home/william/Downloads",
        file_pattern="*.sdpc",
        recursive=False,      # 递归搜索子文件夹
        target_mpp=0.104,
        manual_mpp=0.104,
        padding=1000,
        visualize=True,
        compression='jpeg',
        quality=85,
        tile_size=512,
        bg_threshold=None,
    )