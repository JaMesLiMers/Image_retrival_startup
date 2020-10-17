# --------------------------------------------------------
# By Jameslimer & Aruix
# 用来设定PythonPath, 这样就不需要每次对path进行设定了.
# 用法: 在使用前将该包直接导入即可, 默认执行.
# 因为是根据相对路径进行确定的, 所以请勿改变这个函数的位置, 会有错误.
# --------------------------------------------------------
import os
import sys

def set_python_path():
    """
    设置系统的PythonPath方便包导入

    此文件默认放在utils文件夹下, 该函数使用相对路径来进行path的设置,
    需要的时候直接导入这个包即可. 
    """
    utils_path = os.path.split(os.path.abspath(__file__))[0]
    main_path = os.path.split(utils_path)[0]
    if main_path not in sys.path:
        sys.path.append(main_path[0])

set_python_path()


