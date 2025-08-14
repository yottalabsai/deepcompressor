#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型缓存清理工具

清理下载的模型缓存以释放磁盘空间
"""

import os
import shutil
import sys
from pathlib import Path

def get_cache_directories():
    """获取所有缓存目录"""
    home = os.path.expanduser("~")
    cache_dirs = [
        os.path.join(home, "flux_model_cache"),
        os.path.join(home, ".cache", "model_transfer"),
        os.path.join(home, ".cache", "simple_model_transfer"),
    ]
    return cache_dirs

def get_directory_size(path):
    """计算目录大小"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except (OSError, IOError):
        pass
    return total_size

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"

def list_cached_models():
    """列出所有缓存的模型"""
    print("🔍 扫描缓存的模型...")
    
    cache_dirs = get_cache_directories()
    total_size = 0
    found_models = []
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"\n📁 {cache_dir}:")
            
            try:
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        size = get_directory_size(item_path)
                        if size > 0:
                            found_models.append((item_path, size))
                            total_size += size
                            print(f"  • {item}: {format_size(size)}")
            except PermissionError:
                print(f"  ❌ 无权限访问")
            except Exception as e:
                print(f"  ❌ 错误: {e}")
    
    print(f"\n📊 总计: {len(found_models)} 个模型，{format_size(total_size)}")
    return found_models, total_size

def clean_cache(interactive=True):
    """清理缓存"""
    models, total_size = list_cached_models()
    
    if not models:
        print("✅ 没有找到缓存的模型")
        return
    
    if interactive:
        print(f"\n⚠️  将删除 {len(models)} 个模型，释放 {format_size(total_size)} 空间")
        confirm = input("确认删除所有缓存？(y/N): ")
        if confirm.lower() != 'y':
            print("取消清理")
            return
    
    print("\n🧹 开始清理缓存...")
    
    success_count = 0
    for model_path, size in models:
        try:
            shutil.rmtree(model_path)
            print(f"  ✅ 已删除: {os.path.basename(model_path)} ({format_size(size)})")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 删除失败 {os.path.basename(model_path)}: {e}")
    
    # 清理空的缓存目录
    cache_dirs = get_cache_directories()
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                if not os.listdir(cache_dir):  # 如果目录为空
                    os.rmdir(cache_dir)
                    print(f"  🗑️  删除空目录: {cache_dir}")
            except Exception:
                pass  # 忽略删除空目录的错误
    
    print(f"\n🎉 清理完成！成功删除 {success_count}/{len(models)} 个模型")

def clean_specific_model(model_pattern):
    """清理特定模型"""
    print(f"🔍 搜索包含 '{model_pattern}' 的模型...")
    
    cache_dirs = get_cache_directories()
    found_models = []
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                for item in os.listdir(cache_dir):
                    if model_pattern.lower() in item.lower():
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isdir(item_path):
                            size = get_directory_size(item_path)
                            found_models.append((item_path, size))
                            print(f"  • {item}: {format_size(size)}")
            except Exception as e:
                print(f"❌ 扫描 {cache_dir} 失败: {e}")
    
    if not found_models:
        print(f"❌ 没有找到包含 '{model_pattern}' 的模型")
        return
    
    total_size = sum(size for _, size in found_models)
    print(f"\n📊 找到 {len(found_models)} 个匹配的模型，{format_size(total_size)}")
    
    confirm = input("确认删除这些模型？(y/N): ")
    if confirm.lower() != 'y':
        print("取消删除")
        return
    
    print("\n🧹 开始删除...")
    for model_path, size in found_models:
        try:
            shutil.rmtree(model_path)
            print(f"  ✅ 已删除: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"  ❌ 删除失败: {e}")

def main():
    """主函数"""
    if len(sys.argv) == 1:
        print("🧹 模型缓存清理工具")
        print("=" * 30)
        print("用法:")
        print("  python scripts/clean_model_cache.py list        # 列出缓存的模型")
        print("  python scripts/clean_model_cache.py clean       # 清理所有缓存")
        print("  python scripts/clean_model_cache.py clean <名称> # 清理特定模型")
        print("  python scripts/clean_model_cache.py --force     # 强制清理（无确认）")
        return
    
    action = sys.argv[1].lower()
    
    if action == "list":
        list_cached_models()
    
    elif action == "clean":
        if len(sys.argv) > 2:
            if sys.argv[2] == "--force":
                clean_cache(interactive=False)
            else:
                clean_specific_model(sys.argv[2])
        else:
            clean_cache(interactive=True)
    
    elif action == "--force":
        clean_cache(interactive=False)
    
    else:
        print(f"❌ 未知操作: {action}")
        print("支持的操作: list, clean")

if __name__ == "__main__":
    main()