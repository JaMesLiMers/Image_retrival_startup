import os
import sys
import argparse
import operator
import gc
import functools
import torch
import numpy as np
from dataset.arch_dataset.arch_dataset import Arch

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, default="./dataset/arch_dataset/raw_data", help="Directory that original data in", required=False)
parser.add_argument("-a", "--annotation_file", type=str, default="./dataset/arch_dataset/annotation_data/DemoData_20201228.json", help="path of annotation data", required=False)
parser.add_argument("-o", "--output_dir", type=str, default="./dataset/arch_dataset/processed_data", help="Directory that target data to put", required=False)
args = parser.parse_args()

raw_folder = args.input_dir
processed_folder = args.output_dir
annotation_file = args.annotation_file

os.makedirs(processed_folder, exist_ok=True)

# file name
target_file = "img_data.pt"
target_file_names = "img_names.pt"
output_train_file = "train.pt"
output_test_file = "test.pt"

# full name
full_target_file = os.path.join(raw_folder, target_file)
full_target_file_names = os.path.join(raw_folder, target_file_names)
full_output_train_file = os.path.join(processed_folder, output_train_file)
full_output_test_file = os.path.join(processed_folder, output_test_file)

# handle file not exist error
if not os.path.exists(full_target_file):
    print("Can not find the target file {} in {}".format(target_file, raw_folder))
    exit()
if not os.path.exists(full_target_file_names):
    print("Can not find the target file {} in {}".format(target_file_names, raw_folder))
    exit()
if not os.path.exists(annotation_file):
    print("Can not find the target file {}".format(annotation_file))
    exit()

def main():
    """
    为了回收内存所以包装成了函数
    """
    # load the annotation
    print("loading annotation")
    data_anno_cls = Arch(annotationFile=annotation_file)
    # get all project label and img label
    projects_dict = data_anno_cls.pojToImgs
    img_dict = data_anno_cls.imgs

    # get all target imgs
    targets = []
    concate_list = lambda x: sum([i for i in x.values()], [])
    targets += concate_list(data_anno_cls.filterAnnoLabel("照片"))
    targets += concate_list(data_anno_cls.filterAnnoLabel("效果图"))


    # load dataset 
    print("loading datasets")
    imgs = torch.load(full_target_file)
    img_names = list(torch.load(full_target_file_names))

    # begin processing
    print('Processing...')

    # get index list
    project_list = list(projects_dict.keys())
    project_length = len(project_list)

    # sort for order
    project_list.sort()

    # split index
    all_indexs = list(range(project_length))

    # temp
    data = []
    label = []
    project = []
    class_label = 0
    # counter 
    total_project = len(all_indexs)
    used_project = 0
    min_avliable_img = 3
    # put all train img togeter
    print("processing training data")
    for i in all_indexs:
        # get target class img name list
        project_name = project_list[i]
        image_anno_names_list = projects_dict[project_name]

        # totall img and not found img
        total_img = len(image_anno_names_list)
        not_found_img = 0

        # temp
        _label = []
        _data = []
        for img_anno_name in image_anno_names_list:
            img_id = img_dict[img_anno_name]["imageId"]
            if img_id in targets:
                # get image
                image_name = img_dict[img_anno_name]["fileName"]
                try:
                    image_index = img_names.index(image_name)
                except ValueError:
                    # if not found +1
                    not_found_img += 1 
                    continue
                target_image = imgs[image_index, :, :, :]
                # append
                _label.append(class_label)
                _data.append(target_image)
            else:
                total_img -= 1

        # log result
        # print("class index {} is project {}, total {} imgs, {} imgs not found".format(i, project_name, total_img, not_found_img))
        
        # pass if avaliable image is less than "min_avliable_img"
        if total_img - not_found_img < min_avliable_img or total_img < min_avliable_img:
            # log result
            # print("Project {} avaliable image is less than {}, pass".format(i, min_avliable_img))
            pass
        else:
            label.append(_label)
            data.append(_data)
            project.append("{} - {}".format(class_label,project_name))
            used_project += 1
            class_label += 1
        
        
    # log result for all
    print("Used {} project (total {} project)".format(used_project, total_project))

    print("split to testset and trainset")
    # split index
    split_index = round(used_project * 8 / 10)

    # split to set
    train_dataset = data[0:split_index]
    train_label = label[0:split_index]
    train_project = project[0:split_index]
    # split to set
    test_dataset = data[split_index::]
    test_label = label[split_index::]
    test_project = project[split_index::]

    # convet
    print("converting and saving ...")
    # convert to one list
    reduce_list = lambda x: functools.reduce(operator.concat, x)
    train_dataset = reduce_list(train_dataset)
    train_label = reduce_list(train_label)
    test_dataset = reduce_list(test_dataset)
    test_label = reduce_list(test_label)

    # train convert to tensor and log result
    train_dataset = torch.stack(train_dataset, 0)
    # test convert to tensor and log result
    test_dataset = torch.stack(test_dataset, 0)

    return train_dataset, train_label, train_project, test_dataset, test_label, test_project



# 为了回收内存
train_dataset, train_label, train_project, test_dataset, test_label, test_project = main()

# Saving
print("Used {} image for train".format(len(train_label)))
torch.save([train_dataset, train_label, train_project], full_output_train_file)
del train_dataset, train_label, train_project

print("Used {} image for test".format(len(test_label)))
torch.save([test_dataset, test_label, test_project], full_output_test_file)


print("Done!")
