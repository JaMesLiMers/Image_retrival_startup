# --------------------------------------------------------
# By Jameslimer & Aruix
# 使用YAML进行超参数的管理与配置.
# --------------------------------------------------------

import yaml
import os
import logging


def load_cfg(config_folder, config_file_name):
    """
    to load cfg properties.
    """
    # 获取yaml文件路径
    yml_path = os.path.join(config_folder, config_file_name)
    # open打开文件
    data_file = open(yml_path, 'r', encoding="UTF-8")
    # 读取文件
    cfg_string = data_file.read()
    data_file.close()
    # 用load方法转字典
    cfg = yaml.load(cfg_string, Loader=yaml.FullLoader)
    return cfg
