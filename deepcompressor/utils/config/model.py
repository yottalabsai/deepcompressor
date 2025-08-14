# -*- coding: utf-8 -*-
"""Net configurations."""

import os
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

from omniconfig import configclass

from ..modelscope import is_modelscope_model

__all__ = ["BaseModelConfig"]


@configclass
@dataclass
class BaseModelConfig(ABC):
    """Base class for all model configs.

    Args:
        name (`str`):
            Name of the model.
        family (`str`, *optional*, defaults to `""`):
            Family of the model. If not specified, it will be inferred from the name.
        path (`str`, *optional*, defaults to `""`):
            Path of the model.
        root (`str`, *optional*, defaults to `""`):
            Root directory path for models.
        local_path (`str`, *optional*, defaults to `""`):
            Local path of the model.
        local_root (`str`, *optional*, defaults to `""`):
            Local root directory path for models.
        use_modelscope (`bool`, *optional*, defaults to `False`):
            Whether to use ModelScope for model loading.
        modelscope_cache_dir (`str`, *optional*, defaults to `""`):
            Cache directory for ModelScope models.
    """

    name: str
    family: str = ""
    path: str = ""
    root: str = ""
    local_path: str = ""
    local_root: str = ""
    use_modelscope: bool = False
    modelscope_cache_dir: str = ""

    def __post_init__(self):
        if not self.family:
            self.family = self.name.split("-")[0]
        self.local_root = os.path.expanduser(self.local_root)
        if not self.local_path:
            self.local_path = os.path.join(self.local_root, self.family, self.name)
        if not self.path:
            # 如果启用ModelScope或路径看起来像ModelScope模型ID，则使用ModelScope
            if self.use_modelscope or is_modelscope_model(self.name):
                self.path = self.name  # 使用模型名作为ModelScope模型ID
                self.use_modelscope = True
            else:
                self.path = os.path.join(self.root, self.family, self.name)
        if os.path.exists(self.local_path):
            self.path = self.local_path

    @abstractmethod
    def build(self, *args, **kwargs) -> tp.Any:
        """Build model from config."""
        ...
