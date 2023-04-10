'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


# 点云归一化，转成去心坐标
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


# @@ 最远距离采样
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    # N个点中随机取1个点作为初始采样点
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        # 新增一个采样点，迭代 npoint 次
        centroids[i] = farthest
        # 以采样点为中心点，计算到其他点的距离之和
        centroid = xyz[farthest, :]
        # axis=-1按最高维度雷达，二维则按列加，即行和即x**2+y**2+z**2
        dist = np.sum((xyz - centroid) ** 2, -1)
        # 偏离太远的点置 false 丢弃
        mask = dist < distance
        # 只为为 true 的点更新有效距离， python 允许以bool数组做下标掩膜
        distance[mask] = dist[mask]
        # 取离当前点有效距离最远的点作为下一个采样点
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


# 点云读取、降采样，是否保存降采样的处理结果到文件以避免重复运算
# 返回 (点云，类别标签)，eg. shape((1024,3), number)
class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        # 输入点云采样到多少点
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # rstrip('.com') 删除 string 字符串末尾的指定字符，默认为空白符
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # 存放文件名 table_0001
        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        # 存放类名 table
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # 存放元组 (类名，路径名)
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # + 保存降采样结果，从而不必每次都降采样，加速
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    # 元组 (类名，路径名)
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    # 字典 (类名，从0递增的数字标签)
                    # 与loss函数F.nll_loss相关: http://t.zoukankan.com/leebxo-p-11913939.html
                    cls = np.array([cls]).astype(np.int32)
                    # 点云文件格式: 0.208700,-0.676000,-0.150100,0.203000,-0.943300,0.262700
                    # python 读取txt格式的点云文件，结果为二维数组 point_set = point_set[:, 0:3]
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                # 文件存在直接读取，pickle 为 python 对象序列化工具
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            # eg. label = [12], then label[0] = 12
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        # torch.Size([12, 1024, 3])
        print(point.shape)
        # torch.Size([12])
        print(label.shape)
