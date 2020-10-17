# --------------------------------------------------------
# By Jameslimer & Aruix
# 使用YAML进行超参数的管理与配置.
# --------------------------------------------------------

import yaml
import os
import logging


def load_cfg():
    """
    to load cfg properties.
    """
    # 获取当前脚本所在文件夹路径
    cur_path = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yml_path = os.path.join(cur_path, "data.yml")
    # open打开文件
    data_file = open(yml_path, 'r', encoding="UTF-8")
    # 读取文件
    cfg_string = data_file.read()
    data_file.close()
    # 用load方法转字典
    cfg = yaml.load(cfg_string, Loader=yaml.FullLoader)
    return cfg


cfg = load_cfg()


def init_logger():
    """
    logger cfg
    """
    logger_cfg = cfg.get("log")
    if logger_cfg.get("enable"):
        filename_cfg = os.path.join(os.path.dirname(os.path.realpath(__file__)), logger_cfg.get(
            "file_name"))
        logging.basicConfig(filename=filename_cfg,
                            level=logger_cfg.get("level"))
        logging.info("enable logger")
