# --------------------------------------------------------
# By Jameslimer & Aruix
# 用来计算和保存平均值
# --------------------------------------------------------
import numpy as np

class Meter(object):
    """
    访问AverageMeter中注册过的值后初始化返回的类.

    可以看成是在AverageMeter中注册后生成的描述类.

    Attribute:
        name: 这个类表示的变量的名字
        val: 当前的值
        avg: 当前的平均值(和过去所有的在一起平均)
    """
    def __init__(self, name, val, avg):
        """初始化"""
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        """给程序员的显示接口, 可以直接打Meter的变量来显示"""
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()

class AverageMeter(object):
    """
    计算和保存当前值和平均值

    Attribute:
        val: (dict)key:value保存当前的值.
        sum: (dict)key:value保存累加的总值.
        count: (dict)key:value保存总的batch数量.
    """
    def __init__(self):
        """初始化"""
        self.reset()

    def reset(self):
        """刷新cache"""
        self.val = {}
        self.sum = {} # sum用来存总的数据
        self.count = {}

    def update(self, batch=1, **kwargs):
        """
        更新class中的其中一条或者多条数据

        输入对应的key和value, 算法根据batch的大小先计算平均值, 随后
        加入字典计算的平均值, 用法:
            self.update(key=1)
        如果出现了新的key和value, 则默认初始化他们为0再计算.

        Args:
            batch: 一个batch的大小, 用来计算平均值, 默认为1.
        """
        val = {} 
        # batch norm and update val
        for k in kwargs: 
            val[k] = kwargs[k] / float(batch) 
        self.val.update(val) 
        # update sum & count
        for k in kwargs:
            # for new key & value
            if k not in self.sum:  
                self.sum[k] = 0 
                self.count[k] = 0
            self.sum[k] += kwargs[k] * float(batch)
            self.count[k] += batch

    def __repr__(self):
        """给程序员的显示接口, 可以直接打AverageMeter的变量来显示"""
        s = ''
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        """定义输出的结果和格式"""
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
                    name=attr,
                    val=float(self.val[attr]),
                    avg=float(self.sum[attr]) / self.count[attr])

    def __getattr__(self, attr): 
        """如果访问的属性不再范围内，则调用这个方法. 利用这一特性直接返回字典中已保存的值 """
        # 保证默认行为
        if attr in self.__dict__: 
            return super(AverageMeter, self).__getattr__(attr)
        # 如果没有进行注册的话, 返回0,0(不报错)
        if attr not in self.sum:
            # logger.warn("invalid key '{}'".format(attr))
            # print("invalid key '{}'".format(attr))
            return Meter(attr, 0, 0)
        # 如果进行注册过了, 返回一个简单的类作为展示
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr): # 计算avg
        return float(self.sum[attr]) / self.count[attr]



if __name__ == "__main__":
    """测试平均类的使用(用法例子)"""
    avg = AverageMeter()                    # 初始化
    avg.update(time=1.1, accuracy=.99)      # 传入需要avg的参数
    avg.update(time=1.0, accuracy=.90)      # 多次传入来求平均

    print(avg)           # 将所有的平均值进行打印

    print(avg.time)      # 打印特定的值(str)
    print(avg.time.avg)  # 打印特定的平均(float)
    print(avg.time.val)  # 打印特定的值(float)
    print(avg.Sample)        # 如果出现了没有的默认为0