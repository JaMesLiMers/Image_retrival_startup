import json
import time
import re
import jieba

import numpy as np
import copy
import itertools

import os
from collections import defaultdict
import sys

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve




def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Arch:
    def __init__(self, annotationFile=None, imageFolder=None):
        """
        Constructor of Architecture dataset helper class for reading and visualizing annotations.

        Args:
            annotationFile (str): location of annotation file
            imageFolder (str): location to the folder that hosts images.
        Return:
            None
        """
        # load dataset
        self.dataset,self.anns,self.pojs,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.pojToImgs, self.imgCatToImgs, self.pojCatToPojs = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
        if not annotationFile == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotationFile, 'r', encoding='utf-8'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        """生成需要的index
        
        生成 annotation, project, imgs的字典.

        生成img到project, img到annotation, project到image.

        生成根据标注的映射.
        
        """
        # create index
        print('creating index...')
        anns, pojs, imgs = {}, {}, {}
        imgToAnns,pojToImgs,imgCatToImgs,pojCatToPojs = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
        if 'imageAnnotations' in self.dataset:
            for ann in self.dataset['imageAnnotations']:
                imgToAnns[ann['imageId']].append(ann)
                anns[ann['annotationId']] = ann
                for label in self.extractAllLabel(ann["labels"]):
                    imgCatToImgs[label].append(ann['imageId'])

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['imageId']] = img

        if 'projectAnnotations' in self.dataset:
            for poj in self.dataset['projectAnnotations']:
                pojs[poj['projectId']] = poj
                for label in self.extractLastLabel(poj["projectLabels"]):
                    pojCatToPojs[label].append(poj['projectId'])

        if 'imageAnnotations' in self.dataset and 'projectAnnotations' in self.dataset:
            for ann in self.dataset['imageAnnotations']:
                pojToImgs[ann['projectId']].append(ann['imageId'])


        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.pojToImgs = pojToImgs
        self.imgCatToImgs = imgCatToImgs
        self.pojCatToPojs = pojCatToPojs
        self.imgs = imgs
        self.pojs = pojs

    def extractLastLabel(self, labels):
        """获得最后的一个label

        Args:
            labels: label数据, 例子如下:
            
                [{
                    "label1": label1,
                    "label2": label2,
                    "label3": label3,
                    "label4": label4,
                    "label5": label5,
                }, {
                    "label1": label1,
                    "label2": label2,
                    "label3": label3,
                    "label4": None,
                    "label5": None,
                }]

        Return: 
            一个列表, 里面是所有最后的label
        
        """
        lastLabels = []
        if labels is not None:
            for label in labels:
                for i in range(1, 5):
                    formatString="label{}".format(i)
                    nextFormatString = "label{}".format(i+1)
                    labelContent = label[formatString]
                    next_labelContent = label[nextFormatString]
                    if labelContent is not None and next_labelContent is None:
                        lastLabels.append(labelContent)
                        continue
        return lastLabels

    def extractAllLabel(self, labels):
        """获得所有的label

        Args:
            labels: label数据, 例子如下:
            
                [{
                    "label1": label1,
                    "label2": label2,
                    "label3": label3,
                    "label4": label4,
                    "label5": label5,
                }, {
                    "label1": label1,
                    "label2": label2,
                    "label3": label3,
                    "label4": None,
                    "label5": None,
                }]

        Return: 
            一个列表, 里面是所有的label
        
        """
        lastLabels = []
        if labels is not None:
            for label in labels:
                for i in range(1, 5):
                    formatString="label{}".format(i)
                    nextFormatString = "label{}".format(i+1)
                    labelContent = label[formatString]
                    next_labelContent = label[nextFormatString]
                    if labelContent is not None:
                        lastLabels.append(labelContent)
                    if labelContent is not None and next_labelContent is None:
                        continue
        return lastLabels

    def filterAnnoLabel(self, targetString):
        """过滤需要的label, 只要包含都提取出来

        Args:
            targetString: 要用作搜索的String

        Return:
            所有包含这一标签的img.
        """
        result = {}
        for cat, imgs in self.imgCatToImgs.items():
            if targetString in cat:
                result[cat] = imgs
        return result
    

    def filterProjectLabel(self, targetString):
        """过滤需要的label, 只要包含都提取出来

        Args:
            targetString: 要用作搜索的String

        Return:
            所有包含这一标签的project.
        """
        result = {}
        for cat, projects in self.pojCatToPojs.items():
            if targetString in cat:
                result[cat] = projects
        return result

    def reverseCharForAllContext(self):
        """
        remove all character except english, chinese for "title", "context", "description".
        Store the concate text into dataset.

        Return:
            allContext
        """
        allContext = []
        cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z]")

        for k, v in self.anns.items():
            titleRawText = v["title"] if v["title"] is not None else ""
            contextRawText = v["context"] if v["context"] is not None else ""
            descriptionRawText = v["description"] if v["description"] is not None else ""

            titleCleanText = cop.sub(" ", titleRawText)
            contextCleanText = cop.sub(" ", contextRawText)
            descriptionCleanText = cop.sub(" ", descriptionRawText)

            v["title"] = " ".join(titleCleanText.split())
            v["context"] = " ".join(contextCleanText.split())
            v["description"] = " ".join(descriptionCleanText.split())

            concateText = " ".join((titleCleanText+" "+contextCleanText+" "+descriptionCleanText+"\n").split())
            v["concateText"] = concateText

            # cut all by jieba
            fileSeg = jieba.cut(concateText, cut_all=True)
            fileSeg = list(fileSeg)
            
            v["cutConcateText"] = fileSeg
            fileSeg.append("\n")
                
            allContext.append(" ".join(fileSeg))
        
        return allContext


# Test
if __name__ == "__main__":
    # how to use

    # load dataset
    test = Arch(annotationFile="./Dataset/Arch/DemoData_20201228.json", imageFolder=None)

    # generate context data
    context = test.reverseCharForAllContext()

    # write data into text file
    with open("./Dataset/Arch/concateText.txt", "w") as f:
        f.writelines(context)
