# --------------------------------------------------------
# By Jameslimer & Aruix
# logger的初始化相关函数, logger和对应file_handler的初始化请
# 在主函数中进行.
# --------------------------------------------------------
import os
import logging
import sys
import math

# 用来保存 log 类
logs = set()

def get_format():
    """
    设定log的文件format

    log的格式为:
    [日期 时间 文件名 #行数] 正式信息

    Args:
        None
    
    Returns:
        一个设定好格式的formatter类.
    """
    format_str = '[%(asctime)s-%(filename)s#%(lineno)3d] %(message)s'
    formatter = logging.Formatter(format_str)
    return formatter

def init_log(name, level = logging.INFO, format_func=get_format):
    """
    初始化logger类, 用于全局信息的输出.

    得到的logger类用法如下:
        logger_test.debug('this is a debug log')
        logger_test.info('hello info')
        logger_test.warning('this is a warning log')
        logger_test.error('this is a error message')
        logger_test.critical('this is critical')

    Args:
        name: logger的名称, 如果有的话就直接拉取使用, 如果没有的话就创建一个
            对应名称的logger返回.
        level: 设定log的显示级别, 在这个级别之上的都不予显示, 等级分别有:
            logging.DEBUG
            logging.INFO (默认)
            logging.WARNING
            logging.ERROR
            logging.CRITICAL
        format_func: 设定logger的输出格式, 默认使用上面的函数, 也可以进行自
            定义(自定义的formatter请写在本文件中.)

    Return:
        一个设定好的logger, 默认储存在全局变量logs中, 需要的时候使用同样使用
        该函数, 只要确定确定名称没有错即可取回.
    """
    # In case of already have one
    if (name, level) in logs: 
        return logging.getLogger(name)
    # init logger
    logs.add((name, level))
    logger = logging.getLogger(name) 
    logger.setLevel(logging.DEBUG)          
    # set logging on terminal 
    ch = logging.StreamHandler()     
    ch.setLevel(level)       
    # set format       
    formatter = format_func()
    ch.setFormatter(formatter)       
    logger.addHandler(ch)          
    return logger


def add_file_handler(name, log_file, level=logging.DEBUG):
    """
    为已经初始化好的logger增加指向文件的输出

    让logger不仅可以在命令行中输出信息, 且可以创建指定的文件保存对应的信息, 等
    级过滤可以根据需求设定.

    Args:
        name: 要设定logger的名称, 如果没有的话就先初始化一个.
        log_file: 默认输出到文件的文件目录.
        level: 设定log的显示级别, 在这个级别之上的都不予显示
    """
    logger = logging.getLogger(name)    
    fh = logging.FileHandler(log_file, 'w+')  
    fh.setFormatter(get_format())  
    fh.setLevel(level)
    logger.addHandler(fh)

def print_speed(i, i_time, n, logger_name='global'):
    """
    用来生成目标的进度

    使用方法:
    print_speed(index, index_time, total_index, logger_name)

    Args:
        i: 当前的数量
        i_time: 当前的平均速度
        n: 总共的数量
        logger_name: 要输出的logger名字

    Return: 
        None
    """
    logger = logging.getLogger(logger_name)
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('\nProgress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))


if __name__ == "__main__":
    """
    Usage
    """
    # 生成logger, 一开始自动初始化了一个global logger, 如果需要的话可以重新创建一个不重名的logger
    # 如果已经存在就直接返回创建好的logger
    logger_test= init_log('global', level=logging.INFO)
    # 将logger添加一个handler到文件
    add_file_handler("global", os.path.join('.', 'test.log'), level=logging.INFO)

    # log的方法
    logger_test.debug('this is a debug log')
    logger_test.info('hello info')
    logger_test.warning('this is a warning log')
    logger_test.error('this is a error message')
    logger_test.critical('this is critical')

    # 新增的方法 (默认用global来print)
    print_speed(1, 1, 10)