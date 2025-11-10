import os
import h5py
import numpy as np
from tqdm import tqdm
import sys
import csv
from pathlib import Path

def count_patches_in_dataset(reactive_dir, malt_dir, min_valid_patches=1):
    """
    统计数据集中所有有效的patch数量
    """
    total_patches = 0
    total_files = 0

    def count_from_dir(dir_path, label_name):
        nonlocal total_patches, total_files

        files = [f for f in os.listdir(dir_path) if f.endswith('.h5')]
        dir_total_patches = 0

        print(f"Processing {label_name} files...")
        for f in tqdm(files, desc=f"Counting {label_name}"):
            file_path = os.path.join(dir_path, f)
            try:
                with h5py.File(file_path, 'r') as h5f:
                    if 'features' not in h5f:
                        continue
                    feats = np.array(h5f['features'])
                    if feats.ndim != 2:
                        continue

                    # 统计有效patch数量（非全NaN行）
                    valid_mask = ~np.isnan(feats).all(axis=1)
                    valid_count = valid_mask.sum()

                    if valid_count >= min_valid_patches:
                        dir_total_patches += valid_count
                        total_files += 1

            except Exception as e:
                print(f"⚠️ Skip {file_path}: {e}")

        print(
            f"{label_name} - Files: {len(files)}, Valid files: {total_files if label_name == 'MALT' else total_files - sum(1 for f in os.listdir(reactive_dir) if f.endswith('.h5'))}, Patches: {dir_total_patches}")
        return dir_total_patches

    # 统计两个目录下的patch数量
    reactive_patches = count_from_dir(reactive_dir, "Reactive")
    malt_patches = count_from_dir(malt_dir, "MALT")

    total_patches = reactive_patches + malt_patches

    print(f"\n=== 统计结果 ===")
    print(f"总文件数: {total_files}")
    print(f"总patch数: {total_patches}")
    print(f"Reactive patch数: {reactive_patches}")
    print(f"MALT patch数: {malt_patches}")
    print(f"平均每文件patch数: {total_patches / total_files:.2f}" if total_files > 0 else "无有效文件")

    return total_patches


def inspect_h5_feature(filepath):
    print(f"🔍 Inspecting features in: {filepath}")
    print("=" * 60)

    try:
        with h5py.File(filepath, 'r') as f:
            # 列出所有顶层键
            print("Top-level keys:", list(f.keys()))

            if 'features' not in f:
                print("❌ Error: 'features' dataset not found!")
                return

            features = np.array(f['features'])
            print(f"\n✅ Found 'features' dataset:")
            print(f"   Shape: {features.shape}")
            print(f"   Dtype: {features.dtype}")

            # 检查是否为一维向量（如 768）
            if features.ndim != 1:
                print(f"⚠️  Warning: features is not 1D (ndim={features.ndim})")

            # 统计信息（仅对数值型有效）
            if np.issubdtype(features.dtype, np.number):
                print(f"   Min: {np.min(features):.6f}")
                print(f"   Max: {np.max(features):.6f}")
                print(f"   Mean: {np.mean(features):.6f}")
                print(f"   Std: {np.std(features):.6f}")

                # 检查 NaN / Inf
                nan_count = np.isnan(features).sum()
                inf_count = np.isinf(features).sum()
                print(f"   NaN count: {nan_count}")
                print(f"   Inf count: {inf_count}")

                if nan_count > 0 or inf_count > 0:
                    print("   ⚠️  WARNING: Abnormal values detected!")

            # 打印前10个值（便于人工查看）
            print(f"\n   First 10 values:")
            print("   ", features[:10])

            # 如果太长，也打印最后5个
            if len(features) > 20:
                print(f"   Last 5 values:")
                print("   ", features[-5:])

    except Exception as e:
        print(f"❌ Failed to read {filepath}: {e}")
        sys.exit(1)


def inspect_npy_feature(filepath):
    print(f"🔍 Inspecting features in: {filepath}")
    print("=" * 60)

    try:
        features = np.load(filepath)
        print(f"\n✅ Loaded .npy file:")
        print(f"   Shape: {features.shape}")
        print(f"   Dtype: {features.dtype}")

        # 统计信息（仅对数值型有效）
        if np.issubdtype(features.dtype, np.number):
            print(f"   Min: {np.min(features):.6f}")
            print(f"   Max: {np.max(features):.6f}")
            print(f"   Mean: {np.mean(features):.6f}")
            print(f"   Std: {np.std(features):.6f}")

            # 检查 NaN / Inf
            nan_count = np.isnan(features).sum()
            inf_count = np.isinf(features).sum()
            print(f"   NaN count: {nan_count}")
            print(f"   Inf count: {inf_count}")

            if nan_count > 0 or inf_count > 0:
                print("   ⚠️  WARNING: Abnormal values detected!")

        # 打印前10个值（便于人工查看）
        print(f"\n   First 10 values:")
        print("   ", features.flat[:10] if features.ndim > 1 else features[:10])

        # 如果太长，也打印最后5个
        if features.size > 20:
            print(f"   Last 5 values:")
            print("   ", features.flat[-5:] if features.ndim > 1 else features[-5:])

    except Exception as e:
        print(f"❌ Failed to read {filepath}: {e}")
        sys.exit(1)


def read_coords_legacy(coords_path):
    with h5py.File(coords_path, 'r') as f:
        patch_size = f['coords'].attrs['patch_size']
        patch_level = f['coords'].attrs['patch_level']
        custom_downsample = f['coords'].attrs.get('custom_downsample', 1)
        coords = f['coords'][:]
        print("📋 Legacy Coords Info:")
        print(f"   Patch Size: {patch_size}")
        print(f"   Patch Level: {patch_level}")
        print(f"   Custom Downsample: {custom_downsample}")
        print(f"   Number of Coords: {coords.shape[0]}")
        print(f"   First 5 Coords:\n{coords[:5]}")

def extract_slide_id_from_filename(filename):
    """
    从文件名中提取病理号：取 "-HE" 之前的部分（不包含扩展名）
    例如：S12345-HE.kfb → S12345
    """
    stem = Path(filename).stem  # 去掉扩展名
    if "-HE" in stem:
        slide_id = stem.split("-HE")[0]
        return slide_id
    else:
        return None  # 或者可以 raise ValueError(f"文件名不含 '-HE': {filename}")


def collect_wsi_slide_ids(root_dir_positive, root_dir_negative, output_csv="slide_labels.csv"):
    """
    遍历两个根目录（及其各自的 WSI/HE 子目录），提取病理号并打标签。

    Args:
        root_dir_positive (str): positive 样本的根目录
        root_dir_negative (str): negative 样本的根目录
        output_csv (str): 输出 CSV 文件名
    """
    extensions = {'.sdpc', '.kfb'}
    records = []

    def scan_directory(base_path, label):
        """扫描 base_path 及其 WSI/HE 子目录"""
        base = Path(base_path)
        if not base.exists():
            print(f"警告: 路径不存在: {base_path}")
            return

        # 扫描 base_path 本身
        for file in base.iterdir():
            if file.is_file() and file.suffix.lower() in extensions:
                slide_id = extract_slide_id_from_filename(file.name)
                if slide_id:
                    records.append((slide_id, label))
                else:
                    print(f"跳过无 '-HE' 的文件: {file}")

        # 扫描 WSI/HE 子目录
        he_dir = base / "WSI" / "HE"
        if he_dir.exists() and he_dir.is_dir():
            for file in he_dir.iterdir():
                if file.is_file() and file.suffix.lower() in extensions:
                    slide_id = extract_slide_id_from_filename(file.name)
                    if slide_id:
                        records.append((slide_id, label))
                    else:
                        print(f"跳过无 '-HE' 的文件: {file}")
        else:
            print(f"提示: 未找到 {he_dir} 目录，跳过子目录扫描")

    # 处理 positive 路径
    scan_directory(root_dir_positive, 'positive')
    # 处理 negative 路径
    scan_directory(root_dir_negative, 'negative')

    # 去重：保留首次出现的 slide_id（避免重复标注）
    seen = set()
    unique_records = []
    for slide_id, label in records:
        if slide_id not in seen:
            unique_records.append((slide_id, label))
            seen.add(slide_id)
        else:
            print(f"⚠️ 重复病理号: {slide_id}，已忽略后续出现。")

    # 写入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['slide_id', 'label'])
        writer.writerows(unique_records)

    print(f"✅ 已生成 {output_csv}，共 {len(unique_records)} 条记录。")


if __name__ == "__main__":
    # 使用与ABMIL.py相同的路径
    reactive_dir = "/mnt/gml/GML/Project/Trident/HE/Reactive/20x_224px_0px_overlap/features_virchow"
    malt_dir = "/mnt/gml/GML/Project/Trident/HE/MALT/20x_224px_0px_overlap/features_virchow"

    print("开始统计数据集中的patch数量...")
    count_patches_in_dataset(reactive_dir, malt_dir)