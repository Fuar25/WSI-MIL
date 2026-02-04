from collections import Counter

import re

import os
import pandas as pd
from pathlib import Path


def collect_ihc_files(root_path, output_csv="wsi_paths2.csv"):
    """
    遍历根目录，收集包含 cd3 和 cd20 的 IHC 文件相对路径并保存为 CSV。

    :param root_path: 根目录路径
    :param output_csv: 输出的 CSV 文件名
    """
    root = Path(root_path)
    data = []

    # 定义目标抗体名（转为小写以方便匹配）
    target_antibodies = {'HE'}

    # 遍历根目录下所有的子文件夹（病理号文件夹）
    # 使用 rglob 递归搜索所有 sdpc 文件
    # 路径模式：*/ToRegister/*.sdpc
    search_pattern = "*/ToRegister/*.sdpc"

    for file_path in root.glob(search_pattern):
        # file_path 是一个完整的路径对象
        # 提取文件名，例如: B2018-06208B-cd20.sdpc
        file_name = file_path.name

        # 判断文件名中是否包含目标抗体名
        # 这里使用 any 配合 split 来精准匹配，防止类似 "mcd20" 这种干扰
        if any(antibody in file_name for antibody in target_antibodies):
            # 获取相对于根目录的相对路径
            relative_path = file_path.relative_to(root)
            data.append(str(relative_path))

    # 创建 DataFrame 并保存
    df = pd.DataFrame(data, columns=['wsi'])
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"处理完成！共收集到 {len(df)} 个文件。结果已保存至: {output_csv}")

# 使用示例
# collect_ihc_files("/mnt/gml/GML/DATA/WSI/Reactive-Hyperplasia")


import os
import shutil


def organize_sdpc_files(root_path):
    # 1. 定义分类规则
    # 键是文件夹名称，值是文件后缀的列表
    rules = {
        'HE': ['-HE.sdpc'],
        'CD3-CD20': ['-cd3.sdpc', '-cd20.sdpc'],
        'Ki-67': ['-ki67.sdpc'],
        'CD21': ['-cd21.sdpc'],
        'CK-pan': ['-ckpan.sdpc']
    }

    # 2. 检查根目录是否存在
    if not os.path.exists(root_path):
        print(f"错误：找不到目录 {root_path}")
        return

    # 3. 遍历根目录下的文件
    for filename in os.listdir(root_path):
        file_path = os.path.join(root_path, filename)

        # 只处理文件，跳过文件夹
        if os.path.isfile(file_path):
            target_folder = None

            # 4. 根据规则匹配后缀
            for folder, suffixes in rules.items():
                if any(filename.endswith(s) for s in suffixes):
                    target_folder = folder
                    break

            # 5. 执行移动操作
            if target_folder:
                dest_dir = os.path.join(root_path, target_folder)

                # 如果目标文件夹不存在则创建
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                    print(f"创建文件夹: {target_folder}")

                # 移动文件
                shutil.move(file_path, os.path.join(dest_dir, filename))
                print(f"已移动: {filename} -> {target_folder}/")


import os
import re
import shutil
from pathlib import Path
from typing import Dict, Set

import os
import re
import shutil
from pathlib import Path
from typing import Dict, Set
import pandas as pd


def reorganize_pathology_files(source_root: str, target_root: str):
    """
    将按染色类型组织的病理切片文件重新按病理号和组织区域重组

    参数:
        source_root: 原始根目录路径
        target_root: 目标根目录路径
    """
    # 用于记录每个病理号下的所有区域
    pathology_regions: Dict[str, Set[str]] = {}

    # 正则表达式匹配文件名: (xsB|B)年份-数字+区域字母-染色类型.tiff
    pattern = re.compile(r'^((xsB|B)\d{4}-\d+)([A-Z])-[\w]+\.tiff$')

    print("开始扫描文件...")

    # 第一遍扫描：收集所有文件信息
    file_list = []

    # 遍历原始根目录下的所有子文件夹
    for item in os.listdir(source_root):
        item_path = os.path.join(source_root, item)

        # 只处理文件夹
        if not os.path.isdir(item_path):
            continue

        # 检查是否有 hasROI 子文件夹
        hasroi_path = os.path.join(item_path, 'hasROI')
        if not os.path.exists(hasroi_path):
            continue

        print(f"扫描文件夹: {item}")

        # 扫描 hasROI 文件夹中的 .tiff 文件
        for filename in os.listdir(hasroi_path):
            if not filename.endswith('.tiff'):
                continue

            match = pattern.match(filename)
            if not match:
                print(f"  警告: 文件名格式不匹配 {filename}")
                continue

            pathology_id = match.group(1)  # 病理号
            region = match.group(3)  # 区域字母

            # 记录病理号和区域的对应关系
            if pathology_id not in pathology_regions:
                pathology_regions[pathology_id] = set()
            pathology_regions[pathology_id].add(region)

            # 保存文件信息
            source_file = os.path.join(hasroi_path, filename)
            file_list.append((source_file, filename, pathology_id, region))

    print(f"\n找到 {len(file_list)} 个文件")
    print(f"涉及 {len(pathology_regions)} 个病理号")

    # 创建目标目录结构
    print("\n创建目标目录结构...")
    for pathology_id, regions in pathology_regions.items():
        for region in regions:
            # 创建 Raw 文件夹
            raw_path = os.path.join(target_root, pathology_id, region, 'Raw')
            os.makedirs(raw_path, exist_ok=True)

            # 创建 Reg 文件夹
            reg_path = os.path.join(target_root, pathology_id, region, 'Reg')
            os.makedirs(reg_path, exist_ok=True)

        print(f"  {pathology_id}: 区域 {sorted(regions)}")

    # 移动文件并记录
    print("\n开始移动文件...")
    success_count = 0
    failed_count = 0
    move_records = []

    for source_file, filename, pathology_id, region in file_list:
        target_file = os.path.join(target_root, pathology_id, region, 'Raw', filename)

        try:
            shutil.move(source_file, target_file)
            success_count += 1
            print(f"  ✓ {filename} -> {pathology_id}/{region}/Raw/")

            # 记录成功移动的文件
            move_records.append({
                '文件名': filename,
                '病理号': pathology_id,
                '区域': region,
                '源路径': source_file,
                '目标路径': target_file,
                '状态': '成功'
            })
        except Exception as e:
            failed_count += 1
            print(f"  ✗ 移动失败 {filename}: {e}")

            # 记录失败的文件
            move_records.append({
                '文件名': filename,
                '病理号': pathology_id,
                '区域': region,
                '源路径': source_file,
                '目标路径': target_file,
                '状态': f'失败: {e}'
            })

    print(f"\n完成! 成功: {success_count}, 失败: {failed_count}")

    # 生成Excel记录文件
    if move_records:
        excel_path = os.path.join(target_root, 'file_move_log.xlsx')
        df = pd.DataFrame(move_records)
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\n移动记录已保存到: {excel_path}")


import os
import re
from collections import defaultdict
import pandas as pd


def collect_tiff_by_stain_type(root_dir: str, output_dir: str = None):
    """
    按染色类型统计root目录下的tiff文件，并分别保存为CSV

    参数:
        root_dir: 根目录路径
        output_dir: 输出CSV文件的目录，默认为根目录
    """
    if output_dir is None:
        output_dir = root_dir

    # 用字典存储不同染色类型的文件列表
    stain_files = defaultdict(list)

    # 正则表达式提取染色类型: 文件名格式为 xxx-染色类型.tiff
    pattern = re.compile(r'-(\w+)\.tiff$')

    print(f"开始扫描目录: {root_dir}")

    # 遍历整个目录树
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 只处理.tiff文件
            if not filename.endswith('.tiff'):
                continue

            # 提取染色类型
            match = pattern.search(filename)
            if not match:
                print(f"警告: 无法识别染色类型 - {filename}")
                continue

            stain_type = match.group(1)  # 提取染色类型

            # 获取相对于root的路径
            full_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(full_path, root_dir)

            # 添加到对应染色类型的列表
            stain_files[stain_type].append(relative_path)

    # 为每种染色类型生成CSV文件
    print(f"\n找到 {len(stain_files)} 种染色类型:")

    for stain_type, file_list in sorted(stain_files.items()):
        # 创建DataFrame
        df = pd.DataFrame({'wsi': sorted(file_list)})

        # 生成CSV文件名
        csv_filename = f"{stain_type}_list.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # 保存CSV
        df.to_csv(csv_path, index=False)

        print(f"  {stain_type}: {len(file_list)} 个文件 -> {csv_filename}")

    print(f"\n所有CSV文件已保存到: {output_dir}")

    return stain_files


import os
import shutil
from pathlib import Path


def reorganize_wsi_live(root_path, dry_run=True):
    """
    dry_run=True: 只打印不搬运
    dry_run=False: 真的开始搬运文件并删除空文件夹
    """
    root = Path(root_path)
    if not root.exists():
        print(f"❌ 错误: 找不到根目录 {root_path}")
        return

    target_subfolders = ['Raw', 'Reg']
    moved_count = 0
    deleted_folders = 0

    print(f"--- 状态: {'【模拟模式】' if dry_run else '【🚀 正式执行中】'} ---")

    # 遍历：Patient ID -> Tissue ID
    for tissue_dir in root.glob('*/*'):
        if not tissue_dir.is_dir():
            continue

        for sub_name in target_subfolders:
            sub_path = tissue_dir / sub_name

            if sub_path.exists() and sub_path.is_dir():
                # 1. 搬运文件
                for item in sub_path.iterdir():
                    if item.is_file():
                        dest_path = tissue_dir / item.name

                        if dry_run:
                            print(f"[预览] 移动: {item.name}")
                        else:
                            if not dest_path.exists():
                                # --- 核心搬运逻辑 ---
                                shutil.move(str(item), str(dest_path))
                                moved_count += 1
                            else:
                                print(f"⚠️ 跳过: {item.name} 已存在于 {tissue_dir}")

                # 2. 删除空的 Raw/Reg
                if not dry_run:
                    try:
                        # 再次检查文件夹是否为空（防止有非文件类的子文件夹残留）
                        if not any(sub_path.iterdir()):
                            sub_path.rmdir()
                            deleted_folders += 1
                    except Exception as e:
                        print(f"❌ 无法删除 {sub_path}: {e}")

    print("-" * 40)
    if dry_run:
        print("💡 模拟完成。如果路径看起来没错，请将 dry_run 设置为 False。")
    else:
        print(f"✅ 任务完成！共移动文件: {moved_count} 个，清理文件夹: {deleted_folders} 个。")

import os
from pathlib import Path

def clone_dataset_structure(source_root, target_root):
    """
    完全复刻源目录的文件夹结构到目标路径，不移动/复制任何文件。
    常用于准备存放切片或处理结果的文件夹。
    """
    src = Path(source_root)
    dst = Path(target_root)

    if not src.exists():
        print(f"❌ 错误: 源路径 {source_root} 不存在")
        return

    # 如果目标根目录不存在，则创建
    if not dst.exists():
        dst.mkdir(parents=True)
        print(f"📁 已创建目标根目录: {target_root}")

    dir_count = 0

    # walk 会遍历所有子文件夹
    for root, dirs, files in os.walk(src):
        # 计算相对路径
        relative_path = Path(root).relative_to(src)
        target_dir = dst / relative_path

        # 在目标位置创建对应的文件夹
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            dir_count += 1
            # print(f"已克隆: {relative_path}") # 如果文件夹太多，可以注释掉这行

    print("-" * 40)
    print(f"✅ 结构克隆完成！")
    print(f"源路径: {src.absolute()}")
    print(f"目标路径: {dst.absolute()}")
    print(f"共创建文件夹: {dir_count} 个")


if __name__ == "__main__":
    # # 在这里输入你的根目录路径
    # path = "/mnt/6T/GML/DATA/WSI/SDPC/MALT"
    # organize_sdpc_files(path)

    # folder = "/mnt/6T/GML/DATA/WSI/SDPC/MALT/HE"
    # excel = "/mnt/6T/GML/DATA/WSI/Manifest.xlsx"
    # check_pathology_data(
    #     folder_path= folder,
    #     excel_path= excel,
    #     sheet_name= 'MALT',
    #     target_col= "HE Remarks",
    #     keyword= "1",
    #     check_pairing= False,
    #     suffix=".tiff"
    # )

    # source = "/mnt/6T/GML/DATA/WSI/SDPC/MALT/CD21"
    # target = "/mnt/6T/GML/DATA/WSI/SDPC/MALT/CD21/hasROI"
    # move_paired_files(source, target)

    # # 使用示例
    # source_root = "/mnt/6T/GML/DATA/WSI/SDPC/Reactive"  # 替换为实际的源目录路径
    # target_root = "/mnt/6T/GML/DATA/WSI/Tiff/Reactive"  # 替换为实际的目标目录路径
    #
    # reorganize_pathology_files(source_root, target_root)
    #
    # # 使用示例
    # root_dir = "/mnt/5T/Tiff/MALT"  # 替换为实际的根目录路径
    #
    # # 如果输出到根目录，可以只传一个参数
    # collect_tiff_by_stain_type(root_dir)

    # reorganize_wsi_live('/mnt/5T/GML/Tiff/Reactive', dry_run=False)

    source = "/mnt/5T/GML/Tiff/MALT"  # 你之前的 PatientID 根目录
    target = "/mnt/5T/GML/Reg/MALT"   # 你想要生成空壳的目标目录
    clone_dataset_structure(source, target)
