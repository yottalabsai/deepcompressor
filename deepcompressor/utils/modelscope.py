# -*- coding: utf-8 -*-
"""ModelScope 模型下载和加载工具"""

import os
import typing as tp
from pathlib import Path
from typing import Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from deepcompressor.utils.tools.logging import getLogger

__all__ = ["ModelScopeLoader", "download_model_from_modelscope", "is_modelscope_model"]

logger = getLogger(__name__)


def is_modelscope_model(model_path: str) -> bool:
    """
    判断模型路径是否为ModelScope模型ID
    
    Args:
        model_path: 模型路径或ID
        
    Returns:
        bool: 如果是ModelScope模型ID返回True，否则返回False
    """
    # ModelScope模型ID通常格式为 "organization/model-name" 或 "model-name"
    # 且不是本地路径（不以 ./ 或 / 开头，不包含文件扩展名）
    if not model_path:
        return False
    
    # 如果是本地路径，不是ModelScope模型
    if os.path.exists(model_path) or model_path.startswith(('.//', '/')):
        return False
    
    # 检查是否包含特定的ModelScope标识符或格式
    # ModelScope模型ID可能包含组织名/模型名的格式
    if "/" in model_path and not model_path.startswith("http"):
        # 可能是ModelScope格式，但需要进一步检查
        parts = model_path.split("/")
        if len(parts) == 2 and all(part for part in parts):
            return True
    
    return False


def download_model_from_modelscope(
    model_id: str,
    cache_dir: Union[str, None] = None,
    force_download: bool = False,
    revision: Union[str, None] = None,
) -> str:
    """
    从ModelScope下载模型
    
    Args:
        model_id: ModelScope模型ID
        cache_dir: 缓存目录
        force_download: 是否强制重新下载
        revision: 模型版本
        
    Returns:
        str: 下载后的本地模型路径
    """
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError(
            "ModelScope library is required to download models from ModelScope. "
            "Please install it with: pip install modelscope"
        ) from e
    
    logger.info(f"Downloading model from ModelScope: {model_id}")
    
    # 设置下载参数
    download_kwargs = {
        "model_id": model_id,
        "cache_dir": cache_dir,
        "force_download": force_download,
    }
    
    if revision:
        download_kwargs["revision"] = revision
    
    # 下载模型
    model_path = snapshot_download(**download_kwargs)
    logger.info(f"Model downloaded to: {model_path}")
    
    return model_path


class ModelScopeLoader:
    """ModelScope模型加载器"""
    
    @staticmethod
    def load_llm_model(
        model_id: str,
        cache_dir: Union[str, None] = None,
        torch_dtype: Union[torch.dtype, None] = None,
        device_map: Union[str, None] = None,
        trust_remote_code: bool = True,
        use_fast: bool = True,
        **kwargs
    ) -> tp.Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        从ModelScope加载LLM模型和tokenizer
        
        Args:
            model_id: ModelScope模型ID
            cache_dir: 缓存目录
            torch_dtype: 模型数据类型
            device_map: 设备映射
            trust_remote_code: 是否信任远程代码
            use_fast: 是否使用快速tokenizer
            **kwargs: 其他参数
            
        Returns:
            tuple[PreTrainedModel, PreTrainedTokenizer]: 模型和tokenizer
        """
        try:
            from modelscope import AutoModel, AutoTokenizer as MSAutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ModelScope library is required to load models from ModelScope. "
                "Please install it with: pip install modelscope"
            ) from e
        
        logger.info(f"Loading LLM model from ModelScope: {model_id}")
        
        # 构建模型加载参数
        model_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": trust_remote_code,
        }
        
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype
        if device_map:
            model_kwargs["device_map"] = device_map
        
        # 添加其他kwargs
        model_kwargs.update(kwargs)
        
        # 加载模型
        model = AutoModel.from_pretrained(**model_kwargs)
        
        # 构建tokenizer加载参数
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast,
        }
        
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
        
        # 加载tokenizer
        tokenizer = MSAutoTokenizer.from_pretrained(**tokenizer_kwargs)
        
        logger.info(f"Successfully loaded model and tokenizer from ModelScope: {model_id}")
        return model, tokenizer
    
    @staticmethod
    def load_diffusion_pipeline(
        model_id: str,
        pipeline_class: tp.Type,
        cache_dir: Union[str, None] = None,
        torch_dtype: Union[torch.dtype, None] = None,
        **kwargs
    ):
        """
        从ModelScope加载Diffusion Pipeline
        
        Args:
            model_id: ModelScope模型ID
            pipeline_class: Pipeline类
            cache_dir: 缓存目录
            torch_dtype: 数据类型
            **kwargs: 其他参数
            
        Returns:
            加载的pipeline对象
        """
        # 首先下载模型
        local_model_path = download_model_from_modelscope(
            model_id, cache_dir=cache_dir
        )
        
        # 使用本地路径加载pipeline
        pipeline_kwargs = {
            "torch_dtype": torch_dtype,
        }
        pipeline_kwargs.update(kwargs)
        
        logger.info(f"Loading diffusion pipeline from downloaded model: {local_model_path}")
        pipeline = pipeline_class.from_pretrained(local_model_path, **pipeline_kwargs)
        
        return pipeline