import json

import tqdm  # 进度条显示
import os
import pickle
import logging as log  # 日志记录
import torch
from torch.utils import data
import math
import random


class Dataset(data.Dataset):
    # 初始化方法接受参数problem_number（问题数量）、concept_num（概念数量）、root_dir（数据存储根目录）和可选参数split（数据集分割，默认为'train'）
    def __init__(self, problem_number, concept_num, root_dir, split='train'):
        super().__init__()
        self.map_dim = 0  # 初始化概念映射的维度
        self.prob_encode_dim = 0  # 初始化问题编码向量的维度
        self.path = root_dir  # 将root_dir赋值给实例变量path，表示数据存储根目录
        self.problem_number = problem_number
        self.concept_num = concept_num
        self.show_len = 200  # show_len，表示一个阶段（"show"）的最大长度
        self.split = split  # 数据集分割的类型，可以是'train'、'valid'或'test'
        self.data_list = []  # 初始化空列表存储数据集的样本
        with open("./data/new_mini_09/skill.json", 'r') as load_f:
            self.skill_map = json.load(load_f)
        log.info('Processing data...')
        self.process()
        log.info('Processing data done!')

    def __len__(self):  # 返回数据集的长度，即样本数量
        return len(self.data_list)

    def __getitem__(self, index):  # 根据索引获取数据集中的一个样本
        return self.data_list[index]

    # 用于对一个批次的数据进行整理的方法，接受一个批次的数据作为输入，返回整理后的数据，包括特征数据和标签数据
    def collate(self, batch):
        # seq_num 用于存储序列长度，y 用于存储标签数据, x用于存储特征数据
        seq_num, y = [], []
        x = []
        # 获取一个样本的序列长度
        # batch[0]是第一个样本，[1]表示取样本的特征数据，[1]表示取序列长度
        seq_length = len(batch[0][1][1])
        # 获取一个样本中特征数据的长度,batch[0][1][0]是样本的特征数据，[0][0]表示取特征数据的第一个维度的长度
        x_len = len(batch[0][1][0][0])
        # 遍历序列长度
        for i in range(0, seq_length):
            this_x = []  # 存储当前序列长度的特征数据
            # 遍历特征数据的长度
            for j in range(0, x_len):
                # 将一个空列表添加到 this_x 中，初始化了特征数据列表
                this_x.append([])
                # this_x 添加到 x 中，初始化了一个与序列长度相等的特征数据列表
            x.append(this_x)
            # 遍历批次中的每个样本
        for data in batch:
            # 从当前样本中获取序列长度和特征数据以及标签数据
            this_seq_num, [this_x, this_y] = data
            # 将当前样本的序列长度添加到 seq_num 列表中
            seq_num.append(this_seq_num)
            # 遍历序列长度
            for i in range(0, seq_length):
                # 遍历特征数据的长度
                for j in range(0, x_len):
                    # 将当前样本的特征数据添加到 x 中的相应位置
                    x[i][j].append(this_x[i][j])
            # 将当前样本的标签数据添加到 y 中。由于序列长度可能不等于整个样本的长度，所以只添加序列长度范围内的标签数据
            y += this_y[0: this_seq_num]
        # batch_x 用于存储整理后的特征数据，batch_y 用于存储整理后的标签数据
        batch_x, batch_y = [], []
        # 遍历序列长度
        for i in range(0, seq_length):
            x_info = []  # 存储整理后的特征数据
            # 遍历特征数据的长度
            for j in range(0, x_len):
                # 检查特征数据的索引是否等于2或6
                # 如果索引等于2或6，将特征数据转换为PyTorch张量并添加到 x_info 中
                # 如果索引等于2或6，那么它们表示的是某些离散的数据，如问题ID和相关概念矩阵。这些数据通常是整数类型的，因此需要将它们转换为整数类型的PyTorch张量
                if j == 2 or j == 6:
                    x_info.append(torch.tensor(x[i][j]))
                else:
                    #  # 将特征数据转换为浮点数类型的PyTorch张量并添加到 x_info 中
                    x_info.append(torch.tensor(x[i][j]).float())
                    # 将整理后的特征数据添加到 batch_x 中
            batch_x.append(x_info)
            # 返回整理后的数据。序列长度（seq_num）、特征数据（batch_x）以及标签数据（y）都被转换为PyTorch张量，并作为元组返回。
            # 这个方法的目的是将一个批次的数据整理成适合用于深度学习模型的形式
        return [torch.tensor(seq_num), batch_x], torch.tensor(y).float()

    # 获取用户概念向量的方法，根据相关概念索引和原始用户概念向量，更新用户概念向量
    def get_user_emb(self, related_concept_index, original_user_emb):
        this_user_emb = original_user_emb.copy()
        for concept in related_concept_index:
            if concept == 0:
                # 如果 concept 为0，则跳过本次循环，继续下一个循环
                # 这是因为无相关概念的情况下不需要对用户表示向量进行修改
                continue
                # 如果 concept 不为0，将 this_user_emb 中的对应索引位置的值加1，表示用户与相关概念的联系
                # 这样，最终的 this_user_emb 包含了用户与相关概念的信息
            this_user_emb[concept] += 1
        # 返回修改后的用户表示向量 this_user_emb
        return this_user_emb

    # 获取问题编码向量，将问题ID转化为二进制编码并返回编码向量
    def get_prob_emb(self, problem_id):
        # 将问题ID（problem_id）转化为二进制字符串，并去掉字符串前缀'0b'，得到问题ID的二进制编码
        pro_id_bin = bin(problem_id).replace('0b', '')
        prob_ini_emb = []  # 存储问题的编码向量
        # 遍历问题ID的二进制编码的每个位
        for pro_id_bin_i in pro_id_bin:
            # 初始化变量 appd 为0，用于表示当前位是否为1
            appd = 0
            if pro_id_bin_i == '1':
                appd = 1
            # 将 appd 添加到问题的编码向量 prob_ini_emb 中，表示该位是否为1。
            prob_ini_emb.append(appd)
        # 检查问题的编码向量长度是否小于规定的编码维度
        while len(prob_ini_emb) < self.prob_encode_dim:
            prob_ini_emb = [0] + prob_ini_emb  # 如果长度不足，将0添加到问题的编码向量的前面，扩展编码向量至指定的维度
        # 返回问题的编码向量 prob_ini_emb
        return prob_ini_emb

    # 获取概念编码向量的方法，根据用户掌握的概念列表，将概念编码向量中相应位置设为1
    def get_skill_emb(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    # 获取与用户技能相关的矩阵的方法，构建一个矩阵，表示用户已掌握的概念与其他概念之间的关系
    # 该方法根据用户已掌握的概念列表，构建一个矩阵，其中每行表示一个概念，每列表示一个用户已掌握的概念，矩阵的元素为1表示用户已掌握该概念，0表示未掌握
    def get_related_mat(self, skills):
        skill_mat = []
        for i in skills:
            this_sk = [0] * self.concept_num
            if i != 0:
                this_sk[i] = 1
            skill_mat.append(this_sk)
        return skill_mat

    # 获取用户下一步操作的方法，返回一个向量，表示用户下一步操作
    # 该方法根据用户下一步操作的内容，返回一个向量，其中每个元素表示一种操作，1表示用户将执行该操作，0表示不执行
    def get_after(self, after):
        rt_after = [0] * (self.concept_num - 1)
        for i in after:
            rt_after[i - 1] = 1
        return rt_after

    # 处理学生学习历史数据的方法，接受一个学生的学习历史记录，提取相关信息，构建输入特征和输出标签
    def data_reader(self, stu_records, uid):
        '''
        @params:
            stu_record: learning history of a user
        @returns:
            
        '''
        # x_list 和 y_list 是用来存储特征向量和标签的列表，开始时为空
        x_list = []
        y_list = []
        # last_show列表，每个元素初始化为 300，用于跟踪上次出现的概念或问题。
        last_show = [300] * self.concept_num
        # show_count 是与概念数目相等的列表，开始时所有元素初始化为 0，用于记录每个概念出现的次数
        show_count = [0] * self.concept_num
        # pre_response 用来存储上一次用户的响应（response）的变量，开始时初始化为 0
        pre_response = 0
        # 遍历学生学习历史记录 stu_records 中的每个记录
        for i in range(0, len(stu_records)):
            # 首先从 stu_records 中提取学习历史的信息
            # 包括 order_id（学习历史记录的顺序编号）、problem_id（问题的编号）、skills（用户已掌握的概念列表）和 response（用户的响应）
            order_id, problem_id, skills, response = stu_records[i]
            # 将 skills 转换为概念编码向量 prob，表示用户已掌握的概念
            prob = self.get_skill_emb(skills)
            skill_str = ''
            vv = 0
            for s in skills:
                if s != 0:
                    if vv != 0:
                        skill_str += '_'
                    skill_str += str(s)
                    vv += 1

            skill = 0
            if skill_str in self.skill_map.keys():
                skill = self.skill_map.get(skill_str)
            # 将 problem_id 转换为问题编码向量 prob_bin，用于表示问题的特征
            prob_bin = self.get_prob_emb(problem_id)
            # operate 的列表，开始时初始化为 [1]，如果用户的响应为 0，将 operate 设置为 [0]，这是为了避免除以 0 导致错误
            operate = [1]
            if response == 0:
                operate = [0]  # 避免除0报错
                # 初始化 real_concepts_num 为 0，用于计算用户已掌握的概念数
            real_concepts_num = 0
            # zero_filter 的列表，用于表示概念是否为 0（未掌握)
            zero_filter = []
            # 创建两个名为 last_show_emb 和 show_count_emb 的列表，都初始化为 0，用于构建表示上次出现和出现次数的向量
            last_show_emb = [0] * self.show_len
            show_count_emb = [0] * self.show_len
            # 遍历用户已掌握的概念列表 skills，对于每个概念
            for s in skills:
                if s != 0:
                    # 如果概念不为 0，表示用户已掌握，将 real_concepts_num 增加 1，将 zero_filter 中的对应位置设置为 1，表示已掌握
                    real_concepts_num += 1
                    zero_filter.append(1)
                    # 检查 上次出现的位置last_show 中该概念的值，如果小于 show_len，将 last_show_emb 中的相应位置设置为 1，表示上次出现
                    '''self.show_len 用于跟踪上次出现的位置的最大允许值。如果 last_show[s] 大于或等于 self.show_len，表示该概念最近出现的位置超过了一个阶段的最大长度，
                        所以在特征向量 last_show_emb 和 show_count_emb 中对应的位置被设置为1，表示出现次数超过 self.show_len。这种处理有助于模型捕捉概念的出现和频率信息。'''
                    if last_show[s] < self.show_len:
                        last_show_emb[last_show[s]] = 1
                    # 检查概念出现次数是否不等于0，如果概念出现次数小于 self.show_len，将 show_count_emb 中的相应位置设置为 1，表示出现次数
                    if show_count[s] != 0:
                        if show_count[s] < self.show_len:
                            show_count_emb[show_count[s]] = 1
                        # 如果 show_count 大于 show_len，将 show_count_emb 的最后一个位置设置为 1，表示出现次数超过 show_len
                        else:
                            show_count_emb[self.show_len - 1] = 1
                # 如果概念为 0，将 zero_filter 中的对应位置设置为 0,表示该概念未被掌握。
                else:
                    zero_filter.append(0)
            # 检查用户已掌握的概念数是否为0,如果 real_concepts_num 为 0，将其设置为 1，以避免除以 0 导致错误
            if real_concepts_num == 0:
                real_concepts_num = 1  # 避免除0报错
            # 初始化 related_concept_matrix 为None，这个变量将用于存储与用户技能相关的矩阵
            related_concept_matrix = None
            # 根据已掌握的概念列表长度，决定是否构建 related_concept_matrix
            # 如果已掌握的概念数量小于 5，使用 get_related_mat 方法构建概念关系矩阵，否则将 related_concept_matrix 设置为 0
            if len(skills) < 5:
                related_concept_matrix = self.get_related_mat(skills)
            else:
                related_concept_matrix = 0
# 构建一个包含特征的列表 x_list，这个特征向量包括上次出现、概念编码向量、已掌握的概念列表、概念出现次数、操作、概念是否为 0、问题编号和概念关系矩阵
            x_list.append([
                last_show_emb,
                prob,
                skills,
                show_count_emb,
                operate,
                zero_filter,
                problem_id,
                related_concept_matrix,
                skill,
                uid
            ])
            # 将用户的响应 response 转换为 PyTorch 张量，并添加到标签列表 y_list 中
            y_list.append(torch.tensor(response))
            # 遍历所有概念，用于更新概念的出现次数 show_count 和上次出现的位置 last_show
            # 如果概念不为 0 并且存在于已掌握的概念列表中，将其出现次数加 1，将上次出现的概念设置为 1。如果上次出现的概念不是 300，则将其递增，以跟踪上次出现的位置
            for si in range(0, self.concept_num):
                if si != 0 and si in skills:
                    show_count[si] += 1
                    last_show[si] = 1
                elif last_show[si] != 300:
                    last_show[si] += last_show[s]
#       # 函数返回 x_list 和 y_list，它们将用作训练深度学习模型的输入特征和标签
        return x_list, y_list

    # 处理数据的方法，包括加载学生历史数据，进行数据处理，构建数据集。
    # 这个方法计算问题编码向量的维度，加载学生历史记录，进行数据处理，将处理后的数据添加到数据集中
    def process(self):
        # 打开一个存储学生历史数据的文件
        self.prob_encode_dim = int(math.log(self.problem_number, 2)) + 1
        with open(self.path + 'new_' + self.split + '.pkl', 'rb') as fp:
            histories = pickle.load(fp)
        # 获取学生历史数据的数量
        loader_len = len(histories.keys())
        log.info('loader length: {:d}'.format(loader_len))
        # 初始化一个变量，用于记录已处理的学生历史数据数量
        proc_count = 0
        # 循环遍历学生历史数据
        for k in tqdm.tqdm(histories.keys()):
            # 获取当前学生的学习历史记录
            stu_record = histories[k]
            if stu_record[0] < 10:
                continue
            # 调用data_reader方法处理学生历史记录，将处理后的数据存储在dt中
            dt = self.data_reader(stu_record[1], stu_record[2])
            # 将学生ID和处理后的数据添加到数据集中
            self.data_list.append([stu_record[0], dt])
            proc_count += 1
            # 记录数据处理结束后数据集的最终长度
        log.info('final length {:d}'.format(len(self.data_list)))
