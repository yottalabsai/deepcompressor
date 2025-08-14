#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hugging Face 到 ModelScope 模型迁移脚本

此脚本用于从Hugging Face下载模型，然后上传到ModelScope。
支持各种类型的模型（LLM、Diffusion等）。

使用方法:
python scripts/hf_to_modelscope_transfer.py --hf-model "microsoft/DialoGPT-medium" --ms-model "your-org/DialoGPT-medium"

依赖:
pip install transformers huggingface_hub modelscope gitpython

认证:
1. Hugging Face: huggingface-cli login
2. ModelScope: modelscope login --token "your_token"
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

class ModelTransfer:
    """模型迁移类"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.temp_dir: Optional[str] = None
        
    def __enter__(self):
        """创建持久缓存目录"""
        cache_root = os.path.expanduser("~/.cache/model_transfer")
        os.makedirs(cache_root, exist_ok=True)
        self.temp_dir = cache_root
        self.logger.info(f"使用缓存目录: {self.temp_dir}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """保留缓存目录以供下次使用"""
        if self.temp_dir:
            self.logger.info(f"模型已缓存到: {self.temp_dir}")
            self.logger.info("💡 下次运行相同模型将跳过下载")
    
    def check_dependencies(self):
        """检查依赖库"""
        required_packages = [
            ('transformers', 'transformers'),
            ('huggingface_hub', 'huggingface_hub'),
            ('modelscope', 'modelscope'),
            ('git', 'git')
        ]
        
        missing_packages = []
        
        for package_name, import_name in required_packages:
            try:
                if import_name == 'git':
                    import git
                else:
                    __import__(import_name)
                self.logger.debug(f"✅ {package_name} 已安装")
            except ImportError:
                missing_packages.append(package_name)
                self.logger.error(f"❌ {package_name} 未安装")
        
        if missing_packages:
            install_cmd = f"pip install {' '.join(missing_packages)}"
            if 'git' in missing_packages:
                install_cmd = install_cmd.replace('git', 'gitpython')
            
            raise ImportError(
                f"缺少必要的依赖包: {missing_packages}\n"
                f"请运行: {install_cmd}"
            )
    
    def check_auth(self):
        """检查认证状态"""
        self.logger.info("检查认证状态...")
        
        # 检查Hugging Face认证
        try:
            from huggingface_hub import whoami
            hf_user = whoami()
            self.logger.info(f"✅ Hugging Face已认证: {hf_user.get('name', 'Unknown')}")
        except Exception as e:
            self.logger.warning(f"⚠️  Hugging Face认证检查失败: {e}")
            self.logger.info("提示: 运行 'huggingface-cli login' 进行认证")
        
        # 检查ModelScope认证
        try:
            from modelscope.hub.api import HubApi
            api = HubApi()
            # 尝试获取用户信息来验证认证状态
            try:
                # 这里使用一个简单的API调用来检查认证
                api.list_models(page_size=1)
                self.logger.info("✅ ModelScope已认证")
            except Exception:
                self.logger.warning("⚠️  ModelScope可能未认证")
                self.logger.info("提示: 运行 'modelscope login --token \"your_token\"' 进行认证")
        except ImportError:
            self.logger.error("❌ ModelScope库未安装")
    
    def check_model_exists(self, local_path: str, model_name: str) -> bool:
        """检查模型是否已存在且完整"""
        if not os.path.exists(local_path):
            return False
        
        from pathlib import Path
        path_obj = Path(local_path)
        
        # 检查是否为空目录
        if not any(path_obj.iterdir()):
            return False
        
        # 检查基本文件
        basic_files = ["README.md", "config.json", "model_index.json"]
        found_basic = any((path_obj / f).exists() for f in basic_files)
        
        if not found_basic:
            self.logger.debug(f"未找到基本配置文件")
            return False
        
        # 检查模型文件
        model_files = list(path_obj.rglob("*.safetensors")) + list(path_obj.rglob("*.bin"))
        
        if len(model_files) == 0:
            self.logger.debug(f"未找到模型权重文件")
            return False
        
        # 计算总大小
        total_size = sum(f.stat().st_size for f in path_obj.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        self.logger.info(f"✅ 发现已存在的模型: {size_mb:.2f} MB，{len(model_files)} 个权重文件")
        return True

    def download_from_hf(self, hf_model_name: str, local_path: str, token: Optional[str] = None) -> bool:
        """从Hugging Face下载模型"""
        try:
            from huggingface_hub import snapshot_download
            
            # 检查模型是否已存在
            if self.check_model_exists(local_path, hf_model_name):
                self.logger.info(f"模型已存在，跳过下载: {local_path}")
                return True
            
            self.logger.info(f"开始从Hugging Face下载模型: {hf_model_name}")
            
            download_kwargs = {
                "repo_id": hf_model_name,
                "local_dir": local_path,
                "local_dir_use_symlinks": False,  # 不使用符号链接，确保完整复制
            }
            
            if token:
                download_kwargs["token"] = token
            
            # 下载模型
            snapshot_download(**download_kwargs)
            
            self.logger.info(f"✅ 成功下载模型到: {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 下载失败: {e}")
            return False
    
    def upload_to_modelscope(self, local_path: str, ms_model_name: str, commit_message: str = None) -> bool:
        """上传模型到ModelScope"""
        try:
            from modelscope.hub.api import HubApi
            from modelscope.hub.repository import Repository
            import git
            
            self.logger.info(f"开始上传模型到ModelScope: {ms_model_name}")
            
            # 创建或获取仓库
            api = HubApi()
            
            # 检查仓库是否存在，如果不存在则创建
            try:
                repo_info = api.get_model(ms_model_name)
                self.logger.info(f"找到现有仓库: {ms_model_name}")
            except Exception:
                self.logger.info(f"创建新仓库: {ms_model_name}")
                try:
                    api.create_model(
                        model_id=ms_model_name,
                        visibility=1,  # 公开仓库
                        license='mit',
                        chinese_name=f"从HF迁移的{ms_model_name.split('/')[-1]}模型"
                    )
                except Exception as create_error:
                    self.logger.warning(f"创建仓库失败，可能仓库已存在: {create_error}")
            
            # 创建仓库对象
            repo = Repository(
                local_dir=local_path,
                model_id=ms_model_name,
                clone_from=f"https://www.modelscope.cn/{ms_model_name}.git"
            )
            
            # 检查是否有变更
            repo_git = git.Repo(local_path)
            if repo_git.is_dirty() or repo_git.untracked_files:
                # 添加所有文件
                repo_git.git.add('.')
                
                # 提交变更
                if not commit_message:
                    commit_message = f"Upload model from Hugging Face"
                
                repo_git.index.commit(commit_message)
                
                # 推送到远程
                origin = repo_git.remote('origin')
                origin.push()
                
                self.logger.info("✅ 成功上传模型到ModelScope")
                return True
            else:
                self.logger.info("📝 没有检测到变更，跳过上传")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 上传失败: {e}")
            return False
    
    def verify_model(self, local_path: str) -> dict:
        """验证模型完整性"""
        self.logger.info("验证模型文件...")
        
        model_info = {
            "has_config": False,
            "has_model_files": False,
            "has_tokenizer": False,
            "total_size_mb": 0,
            "file_count": 0
        }
        
        path_obj = Path(local_path)
        
        # 检查重要文件
        important_files = {
            "config.json": "has_config",
            "tokenizer.json": "has_tokenizer",
            "tokenizer_config.json": "has_tokenizer"
        }
        
        model_extensions = ['.bin', '.safetensors', '.pt', '.pth', '.h5']
        
        for file_path in path_obj.rglob('*'):
            if file_path.is_file():
                model_info["file_count"] += 1
                model_info["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                
                # 检查重要文件
                if file_path.name in important_files:
                    model_info[important_files[file_path.name]] = True
                
                # 检查模型文件
                if any(file_path.name.endswith(ext) for ext in model_extensions):
                    model_info["has_model_files"] = True
        
        # 打印验证结果
        self.logger.info("📊 模型验证结果:")
        self.logger.info(f"  - 配置文件: {'✅' if model_info['has_config'] else '❌'}")
        self.logger.info(f"  - 模型文件: {'✅' if model_info['has_model_files'] else '❌'}")
        self.logger.info(f"  - 分词器: {'✅' if model_info['has_tokenizer'] else '❌'}")
        self.logger.info(f"  - 文件总数: {model_info['file_count']}")
        self.logger.info(f"  - 总大小: {model_info['total_size_mb']:.2f} MB")
        
        return model_info
    
    def transfer_model(
        self, 
        hf_model_name: str, 
        ms_model_name: str,
        hf_token: Optional[str] = None,
        commit_message: Optional[str] = None,
        verify_only: bool = False
    ) -> bool:
        """执行完整的模型迁移流程"""
        
        if not self.temp_dir:
            raise RuntimeError("请在with语句中使用此方法")
        
        # 使用模型名称创建更具体的缓存路径
        safe_model_name = hf_model_name.replace("/", "_").replace("-", "_")
        local_path = os.path.join(self.temp_dir, safe_model_name)
        
        # 1. 下载模型
        self.logger.info(f"🚀 开始迁移: {hf_model_name} -> {ms_model_name}")
        
        if not self.download_from_hf(hf_model_name, local_path, hf_token):
            return False
        
        # 2. 验证模型
        model_info = self.verify_model(local_path)
        
        if not model_info["has_model_files"]:
            self.logger.warning("⚠️  未检测到模型文件，可能下载不完整")
        
        if verify_only:
            self.logger.info("🔍 仅验证模式，跳过上传")
            return True
        
        # 3. 上传到ModelScope
        if not self.upload_to_modelscope(local_path, ms_model_name, commit_message):
            return False
        
        self.logger.info(f"🎉 模型迁移完成！")
        self.logger.info(f"   - 源: https://huggingface.co/{hf_model_name}")
        self.logger.info(f"   - 目标: https://modelscope.cn/models/{ms_model_name}")
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从Hugging Face下载模型并上传到ModelScope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 基本迁移:
   python scripts/hf_to_modelscope_transfer.py \\
     --hf-model "microsoft/DialoGPT-medium" \\
     --ms-model "your-org/DialoGPT-medium"

2. 使用自定义token:
   python scripts/hf_to_modelscope_transfer.py \\
     --hf-model "private-org/private-model" \\
     --ms-model "your-org/private-model" \\
     --hf-token "hf_xxxx"

3. 仅验证模型（不上传）:
   python scripts/hf_to_modelscope_transfer.py \\
     --hf-model "microsoft/DialoGPT-medium" \\
     --ms-model "your-org/DialoGPT-medium" \\
     --verify-only

4. 批量迁移:
   # 创建模型列表文件 models.txt
   echo "microsoft/DialoGPT-small your-org/DialoGPT-small" >> models.txt
   echo "microsoft/DialoGPT-medium your-org/DialoGPT-medium" >> models.txt
   
   # 批量处理
   while read hf_model ms_model; do
     python scripts/hf_to_modelscope_transfer.py --hf-model "$hf_model" --ms-model "$ms_model"
   done < models.txt

注意事项:
- 确保已通过 'huggingface-cli login' 认证Hugging Face
- 确保已通过 'modelscope login --token "xxx"' 认证ModelScope
- 大模型下载可能需要较长时间，请耐心等待
- 建议先使用 --verify-only 选项验证模型完整性
        """
    )
    
    parser.add_argument(
        "--hf-model", 
        required=True,
        help="Hugging Face模型名称 (例如: microsoft/DialoGPT-medium)"
    )
    
    parser.add_argument(
        "--ms-model", 
        required=True,
        help="ModelScope模型名称 (例如: your-org/DialoGPT-medium)"
    )
    
    parser.add_argument(
        "--hf-token",
        help="Hugging Face访问token（用于私有模型）"
    )
    
    parser.add_argument(
        "--commit-message",
        help="提交信息（默认: 'Upload model from Hugging Face'）"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅下载和验证模型，不上传到ModelScope"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 开始迁移
    start_time = time.time()
    
    try:
        with ModelTransfer(logger) as transfer:
            # 检查依赖和认证
            transfer.check_dependencies()
            transfer.check_auth()
            
            # 执行迁移
            success = transfer.transfer_model(
                hf_model_name=args.hf_model,
                ms_model_name=args.ms_model,
                hf_token=args.hf_token,
                commit_message=args.commit_message,
                verify_only=args.verify_only
            )
            
            elapsed_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ 迁移成功完成！耗时: {elapsed_time:.2f}秒")
                return 0
            else:
                logger.error(f"❌ 迁移失败！耗时: {elapsed_time:.2f}秒")
                return 1
                
    except KeyboardInterrupt:
        logger.info("⏹️  用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"💥 意外错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main())