#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSI文件收集与恢复脚本
用于将分散在各病理号文件夹下ToRegister中的WSI文件集中处理后恢复
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
import argparse


class WSIFileManager:
    """WSI文件管理器"""

    def __init__(self, root_dir: str, target_dir: str, mapping_file: str = "wsi_mapping.json"):
        """
        初始化

        Args:
            root_dir: 根目录,包含各病理号文件夹
            target_dir: 统一处理目录
            mapping_file: 映射文件路径
        """
        self.root_dir = Path(root_dir)
        self.target_dir = Path(target_dir)
        self.mapping_file = Path(mapping_file)
        self.wsi_extensions = ['.svs', '.sdpc', '.kfb']

    def collect_files(self) -> Dict[str, str]:
        """
        收集所有WSI文件到统一目录

        Returns:
            文件映射字典 {文件名: 原始路径}
        """
        if not self.root_dir.exists():
            raise FileNotFoundError(f"根目录不存在: {self.root_dir}")

        # 创建目标目录
        self.target_dir.mkdir(parents=True, exist_ok=True)

        file_mapping = {}
        collected_count = 0

        print(f"开始从 {self.root_dir} 收集WSI文件...")

        # 遍历所有病理号文件夹
        for pathology_dir in self.root_dir.iterdir():
            if not pathology_dir.is_dir():
                continue

            # 检查ToRegister文件夹
            to_register_dir = pathology_dir / "ToRegister"
            if not to_register_dir.exists():
                continue

            # 查找WSI文件
            for file_path in to_register_dir.iterdir():
                if file_path.suffix.lower() in self.wsi_extensions:
                    # 保持原文件名不变
                    filename = file_path.name
                    new_path = self.target_dir / filename

                    # 检查目标是否已存在(理论上SDPC不会重复,但SVS可能会)
                    if new_path.exists():
                        print(f"  ⚠ 警告: 文件已存在 {filename}, 跳过")
                        continue

                    # 移动文件
                    try:
                        shutil.move(str(file_path), str(new_path))
                        file_mapping[filename] = str(file_path.relative_to(self.root_dir))
                        collected_count += 1
                        print(f"  ✓ 收集: {filename}")
                    except Exception as e:
                        print(f"  ✗ 错误: 移动 {filename} 失败 - {e}")

        # 保存映射关系
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(file_mapping, f, ensure_ascii=False, indent=2)

        print(f"\n收集完成! 共收集 {collected_count} 个文件")
        print(f"映射文件已保存至: {self.mapping_file}")

        return file_mapping

    def restore_files(self, mode: str = 'simple') -> int:
        """
        根据映射文件恢复所有文件到原始位置

        Args:
            mode: 恢复模式
                - 'simple': 只恢复原始文件
                - 'with-results': 恢复原始文件,并检查同名.svs结果文件一起恢复

        Returns:
            恢复的文件数量
        """
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"映射文件不存在: {self.mapping_file}")

        # 读取映射关系
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            file_mapping = json.load(f)

        restored_count = 0

        print(f"开始恢复文件到原始位置 (模式: {mode})...")

        for filename, relative_original_path in file_mapping.items():
            source_path = self.target_dir / filename
            target_path = self.root_dir / relative_original_path

            if not source_path.exists():
                print(f"  ⚠ 警告: 文件不存在 {filename}")
                continue

            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 移动原始文件回原位
            try:
                shutil.move(str(source_path), str(target_path))
                restored_count += 1
                print(f"  ✓ 恢复: {filename}")
            except Exception as e:
                print(f"  ✗ 错误: 恢复 {filename} 失败 - {e}")
                continue

            # with-results模式: 检查是否有同名.svs结果文件
            if mode == 'with-results':
                # 获取不带扩展名的文件名
                file_stem = Path(filename).stem

                # 查找同名的.svs文件(如果原文件不是.svs)
                if not filename.lower().endswith('.svs'):
                    result_filename = f"{file_stem}.svs"
                    result_source = self.target_dir / result_filename

                    if result_source.exists():
                        result_target = target_path.parent / result_filename
                        try:
                            shutil.move(str(result_source), str(result_target))
                            restored_count += 1
                            print(f"    ↳ 同时恢复结果文件: {result_filename}")
                        except Exception as e:
                            print(f"    ✗ 错误: 恢复结果文件 {result_filename} 失败 - {e}")

        print(f"\n恢复完成! 共恢复 {restored_count} 个文件")

        return restored_count

    def list_files(self) -> List[str]:
        """列出当前映射的所有文件"""
        if not self.mapping_file.exists():
            print("映射文件不存在")
            return []

        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            file_mapping = json.load(f)

        print(f"\n映射文件中共有 {len(file_mapping)} 个文件:")
        for filename, orig_path in file_mapping.items():
            print(f"  {filename} <- {orig_path}")

        return list(file_mapping.keys())


def main():
    parser = argparse.ArgumentParser(description='WSI文件收集与恢复工具')
    parser.add_argument('action', choices=['collect', 'restore', 'list'],
                        help='操作: collect(收集), restore(恢复), list(列表)')
    parser.add_argument('--root', required=True, help='根目录路径')
    parser.add_argument('--target', default='./wsi_processing', help='统一处理目录')
    parser.add_argument('--mapping', default='wsi_mapping.json', help='映射文件路径')
    parser.add_argument('--mode', choices=['simple', 'with-results'], default='simple',
                        help='恢复模式: simple(仅原文件), with-results(含结果文件)')

    args = parser.parse_args()

    manager = WSIFileManager(
        root_dir=args.root,
        target_dir=args.target,
        mapping_file=args.mapping
    )

    if args.action == 'collect':
        manager.collect_files()
    elif args.action == 'restore':
        manager.restore_files(mode=args.mode)
    elif args.action == 'list':
        manager.list_files()


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("WSI文件收集与恢复工具")
    print("=" * 60)
    print("\n使用方法:")
    print("1. 收集文件:")
    print("   python script.py collect --root /path/to/root")
    print("\n2. 恢复文件(仅原文件):")
    print("   python script.py restore --root /path/to/root --mode simple")
    print("\n3. 恢复文件(包含同名.svs结果文件):")
    print("   python script.py restore --root /path/to/root --mode with-results")
    print("\n4. 查看映射:")
    print("   python script.py list --root /path/to/root")
    print("=" * 60)
    
    # 如果直接运行,显示帮助信息
    import sys

    sys.argv = [
        "get_sdpc.py",
        'collect',
        '--root', '/mnt/6T/GML/DATA/WSI/MALT-Lymphoma',
        '--target', '/mnt/6T/GML/DATA/WSI/SDPC/MALT',
        '--mapping', '/mnt/6T/GML/DATA/WSI/SDPC/MALT/malt.json'
    ]

    if len(sys.argv) == 1:
        sys.exit(0)

    main()