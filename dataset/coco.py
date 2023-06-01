import os
import os.path as osp
import numpy as np
import random
import cv2
from collections import defaultdict

from .base_dataset import BaseDataset

class GFSSegTrain(BaseDataset):
    num_classes = 80
    def __init__(self, root, list_path, fold, shot=1, mode='train', crop_size=(512, 512),
             ignore_label=255, base_size=(2048,512), resize_label=False, filter=False, seed=123):
        super(GFSSegTrain, self).__init__(mode, crop_size, ignore_label, base_size=base_size)
        assert mode in ['train', 'val_supp']
        self.root = root
        self.list_path = list_path
        self.fold = fold
        self.shot = shot
        self.mode = mode
        self.resize_label = resize_label
        self.ratio_range = (0.8, 1.25)
        self.img_dir = 'train2014'
        self.lbl_dir = 'train_no_crowd'

        if fold == -1:
            # training with all classes
            self.base_classes = set(range(1, self.num_classes+1))
            self.novel_classes = set()
            with open(os.path.join(self.list_path), 'r') as f:
                self.data_list = f.read().splitlines()
        else:
            # base classes
            self.base_classes = set(range(1, self.num_classes + 1)) - set(range(fold + 1, fold + 78, 4))
            # novel classes
            self.novel_classes = set(range(fold + 1, fold + 78, 4))

            filter_flag = True if (self.mode == 'train' and filter) else False
            list_dir = os.path.dirname(self.list_path)
            list_dir = list_dir + '/fold%s'%fold
            if filter_flag:
                list_dir = list_dir + '_filter'
            list_saved = os.path.exists(os.path.join(list_dir, 'train_fold%s.txt'%fold))
            if list_saved:
                print('id files exist...')
                self.novel_cls_to_ids = defaultdict(list)
                with open(os.path.join(list_dir, 'train_fold%s.txt'%fold), 'r') as f:
                    self.data_list = f.read().splitlines()
                for cls in self.novel_classes:
                    with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'r') as f:
                        self.novel_cls_to_ids[cls] = f.read().splitlines()
                with open(os.path.join(list_dir, 'fold%s_%sshot_seed%s.txt'%(fold, shot, seed)), 'r') as f:
                        self.novel_id_list = f.read().splitlines()
            else:
                '''
                fold0/train_fold0.txt: training images containing base classes (novel classes will be ignored during training)
                fold0/train_novel_class[0-4].txt: training images containing novel class [0-4] (to provide support images for validation)
                '''
                with open(os.path.join(self.list_path), 'r') as f:
                    self.ids = f.read().splitlines()
                print('checking ids...')
                
                self.data_list, self.novel_cls_to_ids, self.base_cls_to_ids = self._filter_and_map_ids(filter_intersection=filter_flag)
                if not os.path.exists(list_dir):
                    os.makedirs(list_dir)
                with open(os.path.join(list_dir, 'train_fold%s.txt'%fold), 'w') as f:
                    for id in self.data_list:
                        f.write(id+"\n")
                for cls in self.novel_classes:
                    with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'w') as f:
                        for id in self.novel_cls_to_ids[cls]:
                            f.write(id+"\n")
                for cls in self.base_classes:
                    with open(os.path.join(list_dir, 'train_base_class%s.txt'%cls), 'w') as f:
                        for id in self.base_cls_to_ids[cls]:
                            f.write(id+"\n")

    def __len__(self):
        if self.mode == 'val_supp':
            return len(self.novel_classes)
        else:
            return len(self.data_list)

    def _convert_label(self, label):
        new_label = label.copy()
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        assert len(label_class) > 0
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)
        for c in label_class:
            if c in base_list:
                new_label[label == c] = (base_list.index(c) + 1)    # 0 as background
            elif c in novel_list:
                if self.mode == 'train':
                    new_label[label == c] = 0
                else:
                    new_label[label == c] = (novel_list.index(c) + len(base_list) + 1)
        return new_label

    def __getitem__(self, index):
        if self.mode == 'val_supp':
            return self._get_val_support(index)
        else:
            return self._get_train_sample(index)

    def _get_train_sample(self, index):
        id = self.data_list[index]
        image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id), cv2.IMREAD_COLOR)
        label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id), cv2.IMREAD_GRAYSCALE)

        label = self._convert_label(label)
        # date augmentation & preprocess
        image, label = self.resize(image, label, random_scale=True)
        image, label = self.random_flip(image, label)
        image, label = self.crop(image, label)
        image = self.normalize(image)
        image, label = self.pad(self.crop_size, image, label)
        image, label = self.totensor(image, label)

        return image, label, id

    def _get_val_support(self, index):
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)

        target_cls = novel_list[index]
        novel_cls_id = index + len(base_list) + 1

        id_s_list, image_s_list, label_s_list = [], [], []

        for k in range(self.shot):
            id_s = self.novel_id_list[index*self.shot+k]              
            image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id_s), cv2.IMREAD_COLOR)
            label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id_s), cv2.IMREAD_GRAYSCALE)

            label = self._convert_label(label)
            # date augmentation & preprocess
            image, label = self.resize(image, label)
            image = self.normalize(image)
            image, label = self.pad(self.crop_size, image, label)
            image, label = self.totensor(image, label)
            id_s_list.append(id_s)
            image_s_list.append(image)
            label_s_list.append(label)
        # print(target_cls)
        # print(id_s_list)
        return image_s_list, label_s_list, id_s_list, novel_cls_id

    def _filter_and_map_ids(self, filter_intersection=False):
        image_label_list = []
        base_cls_to_ids = defaultdict(list)
        novel_cls_to_ids = defaultdict(list)
        for i in range(len(self.ids)):
            mask = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%self.ids[i]), cv2.IMREAD_GRAYSCALE)
            label_class = np.unique(mask).tolist()
            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255)
            valid_base_classes = set(np.unique(mask)) & self.base_classes
            valid_novel_classes = set(np.unique(mask)) & self.novel_classes

            if valid_base_classes:
                new_label_class = []
                if filter_intersection:
                    if set(label_class).issubset(self.base_classes):
                        image_label_list.append(self.ids[i])
                        for cls in valid_base_classes:
                            if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                                new_label_class.append(cls)

                else:
                    image_label_list.append(self.ids[i])
                    for cls in valid_base_classes:
                        if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                            new_label_class.append(cls)
                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        base_cls_to_ids[cls].append(self.ids[i])

            if valid_novel_classes:
            # remove images whose valid objects are all small (according to PFENet)
                new_label_class = []
                for cls in valid_novel_classes:
                    if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                        new_label_class.append(cls)

                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        novel_cls_to_ids[cls].append(self.ids[i])

        return image_label_list, novel_cls_to_ids, base_cls_to_ids

class GFSSegVal(BaseDataset):
    num_classes = 80
    def __init__(self, root, list_path, fold, crop_size=(512, 512),
             ignore_label=255, base_size=(2048,512), resize_label=False, use_novel=True, use_base=True):
        super(GFSSegVal, self).__init__('val', crop_size, ignore_label, base_size=base_size)
        self.root = root
        self.list_path = list_path
        self.fold = fold
        self.resize_label = resize_label
        self.use_novel = use_novel
        self.use_base = use_base
        self.img_dir = 'val2014'
        self.lbl_dir = 'val_no_crowd'

        # base classes
        self.base_classes = set(range(1, self.num_classes + 1)) - set(range(fold + 1, fold + 78, 4))
        # novel classes
        self.novel_classes = set(range(fold + 1, fold + 78, 4))

        with open(os.path.join(self.list_path), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id), cv2.IMREAD_COLOR)
        label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id), cv2.IMREAD_GRAYSCALE)

        new_label = label.copy()
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)

        for c in label_class:
            if c in base_list:
                if self.use_base:
                    new_label[label == c] = (base_list.index(c) + 1)    # 0 as background
                else:
                    new_label[label == c] = 0
            elif c in novel_list:
                if self.use_novel:
                    if self.use_base:
                        new_label[label == c] = (novel_list.index(c) + len(base_list) + 1)
                    else:
                        new_label[label == c] = (novel_list.index(c) + 1)
                else:
                    new_label[label == c] = 0

        label = new_label.copy()

        # date augmentation & preprocess
        if self.resize_label:
            image, label = self.resize(image, label)
            image = self.normalize(image)
            image, label = self.pad(self.base_size, image, label)
        else:
            image = self.resize(image)
            image = self.normalize(image)
            image = self.pad(self.base_size, image)
        image, label = self.totensor(image, label)

        return image, label, id
