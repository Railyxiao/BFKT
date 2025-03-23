import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA  # 从sklearn库中导入PCA（主成分分析）类，用于数据降维
import math
import modules
from torch.autograd import Variable
import pickle
import random

from bertmodel import BERT


class iekt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.node_dim = args.dim
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        # 创建一个名为"predictor"的神经网络模型，其结构由modules.funcs类定义，具有多层感知机（MLP）结构。它的输入维度是"args.dim * 5"，输出维度为1
        self.predictor = modules.funcs(args.n_layer, args.dim * 10, 1, args.dropout)
        # 创建两个可训练的参数矩阵："cog_matrix"和"acq_matrix"，它们的初始化值是随机的，
        # 维度分别为"args.cog_levels x args.dim * 2"和"args.acq_levels x args.dim * 2"
        self.cog_matrix = nn.Parameter(torch.randn(args.cog_levels, args.dim * 2).to(args.device), requires_grad=True)
        self.acq_matrix = nn.Parameter(torch.randn(args.acq_levels, args.dim * 2).to(args.device), requires_grad=True)
        # 创建两个名为"select_preemb"和"checker_emb"的神经网络模型，其结构也由modules.funcs类定义，用于计算概率。这些模型接受输入数据，经过多层感知机后输出概率
        self.select_preemb = modules.funcs(args.n_layer, args.dim * 3, args.cog_levels, args.dropout)
        self.checker_emb = modules.funcs(args.n_layer, args.dim * 12, args.acq_levels, args.dropout)
        # 创建参数矩阵"prob_emb",维度为"(args.problem_number - 1) x args.dim"用于表示问题的嵌入
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number - 1, args.dim).to(args.device), requires_grad=True)
        self.gru_h = modules.mygru(0, args.dim * 4, args.dim)
        showi0 = []
        for i in range(0, args.n_epochs):
            showi0.append(i)
            # 创建一个名为"show_index"的张量，其值是一个从0到"args.n_epochs - 1"的整数序列，用于表示一个时间步的索引
        self.show_index = torch.tensor(showi0).to(args.device)  # self.show_index=(1,300)
        # concept_emb维度为"(self.concept_num - 1) x args.dim",用于表示概念的嵌入
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num - 1, args.dim).to(args.device), requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()
        # Attention Mechanism Parameters
        self.attention = nn.Linear(args.dim * 4, 1)
        self.softmax = nn.Softmax(dim=1)
        self.akt = modules.AKT(n_question=143, n_pid=args.problem_number, n_blocks=1, d_model=256,
                               dropout=0.05, kq_same=1, model_type='akt', l2=1e-05).to(args.device)
        self.d_output = None
        self.akt_output = None
        with open("./data/new_mini_09/question-rate.json", 'r') as load_f:
            self.question_map = json.load(load_f)
        self.question_model = modules.improved_ce()
        self.bert_model = BERT(
            hidden_size=640,
            num_head=16,
            num_encoder=12,
            max_seq_len=200,
            dropout_p=.1,
            use_leakyrelu=True
        ).to(args.device)

    # 获取问题的表示Vt向量(将问题的嵌入向量计算为概念级别表示和问题嵌入的拼接，并返回这个表示作为输出)
    '''它是一个拼接了概念级别的表示和问题嵌入的向量。概念级别的表示是根据问题涉及的相关概念和过滤器计算的，
    问题嵌入是根据问题编号从prob_emb参数矩阵中提取的。这个函数的输入是prob_ids, related_concept_index, 
    filter0, data_len，分别表示问题编号、相关概念索引、过滤器张量、数据长度。这个函数的输出是v，也就是问题的表示。'''

    def get_ques_representation(self, prob_ids, related_concept_index, filter0, data_len):  # 问题标识符,相关概念索引，过滤器的张量，数据长度
        # 创建一个包含零向量的张量"concepts_cat"，并将其复制成"data_len"个副本,concepts_cat其中包含了零向量以及概念嵌入（self.concept_emb）。
        # 这个张量将被用于提取相关概念
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
             self.concept_emb],
            dim=0).unsqueeze(0).repeat(data_len, 1,
                                       1)  # (11,80,64), 11 个问题中，每个问题有 80 个概念，而每个概念的嵌入是一个长度为 64 的向量(data_len=11)
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)
        # 从"concepts_cat"中提取相关概念，这些相关概念由 related_concept_index 和 filter0 进行筛选，并计算概念级别的表示。
        # 这将生成名为 concept_level_rep 的概念级别表示
        related_concepts = concepts_cat[r_index, related_concept_index, :]  # (11,4,64)
        filter_sum = torch.sum(filter0, dim=1)  # filter_sum=(1,11) filter0=(11,4)
        # 计算一个用于后续计算的 div 张量，该张量确保了在除法操作中不会出现除数为零的情况
        div = torch.where(filter_sum == 0,
                          torch.tensor(1.0).to(self.device),
                          filter_sum
                          ).unsqueeze(1).repeat(1, self.node_dim)

        concept_level_rep = torch.sum(related_concepts, dim=1) / div  # (11,64) 表示每个数据点的概念级别表示
        # prob_cat 是用来存储问题嵌入（question embeddings）以及一个特殊标记的 PyTorch 张量
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb],
            dim=0)  # (1766,64) prob_cat 是一个张量，其维度是 (num_probs + 1, self.node_dim)，其中 num_probs 是问题嵌入的数量。
        # 它包含了一个全零向量作为标记，紧接着是问题嵌入的向量。这种操作通常用于构建问题嵌入的词典，以便能够在索引问题时添加一个特殊标记
        # 创建一个包含问题嵌入的张量"item_emb"，根据"prob_ids"提取问题的嵌入
        item_emb = prob_cat[prob_ids]  # (11,64)

        # Calculate attention weights for question embeddings
        # attention_weights = self.softmax(self.attention(item_emb))
        #
        # # Calculate the attended question embeddings
        # attended_question_emb = torch.mul(attention_weights, item_emb)
        # 将概念级别的表示和问题嵌入拼接成一个表示张量"v"，然后返回
        v = torch.cat(
            [concept_level_rep,
             item_emb],
            dim=1)  # (11,128)
        return v

    # 计算认知选择概率fp
    # 接受一个张量"x"和一个"softmax_dim"参数，用于指定softmax操作的维度。方法通过对输入的"x"应用softmax函数来计算概率，然后返回结果
    def pi_cog_func(self, x, softmax_dim=1):  # x=(11,192)
        return F.softmax(self.select_preemb(x), dim=softmax_dim)

    # 获取问题的表示"v"
    def obtain_v(self, this_input, h, x, emb, seqi):
        # 从"this_input"中提取各种输入数据
        last_show, problem, related_concept_index, show_count, operate, filter0, prob_ids, related_concept_matrix, skill, uid = this_input

        data_len = operate.size()[0]  # data_len=11

        emb = self.improve_emb(emb, related_concept_index, len(related_concept_index))
        # 调用"get_ques_representation"方法获取问题表示"v"
        v = self.get_ques_representation(prob_ids, related_concept_index, filter0, data_len)
        # 将"h"和"v"拼接成一个新的张量"h_v"
        predict_x = torch.cat([h, v], dim=1)  # (11,192)
        h_v = torch.cat([h, v], dim=1)  # (11,192)
        # 使用"h_v"和"emb"作为输入，通过"predictor"模型计算问题概率"prob"
        cat_s = torch.cat([
            predict_x, emb, self.d_output[:, seqi]
        ], dim=1)
        prob = self.predictor(cat_s)  # (11,1)
        return h_v, v, prob, x  # v=(),x=(11,1,128)

    # 更新模型状态,这个 update_state 函数用于基于当前的隐藏状态 h 和其他输入参数来计算下一个隐藏状态 next_p_state
    def update_state(self, h, v, emb, operate):
        # 基于输入参数计算两个张量"v_cat"和"e_cat"，将这些张量拼接成一个新的输入张量"inputs"
        # 它的作用是根据学生的回答结果来调整问题表示，使其更能反映问题的难度和相关性
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.node_dim * 2)),
            v.mul((1 - operate).repeat(1, self.node_dim * 2))], dim=1)  # (11,256)
        # e_cat的作用是根据学生的回答结果来调整嵌入数据，使其更能反映学生的认知水平和知识获取敏感性。
        e_cat = torch.cat([
            emb.mul((1 - operate).repeat(1, self.node_dim * 2)),
            emb.mul((operate).repeat(1, self.node_dim * 2))], dim=1)  # (11,256)
        inputs = v_cat + e_cat
        # 使用"gru_h"模型更新隐藏状态"h"，并返回下一个状态"next_p_state"
        next_p_state = self.gru_h(inputs, h)  # (11,64)
        return next_p_state

    # 用于计算感知选择概率
    def pi_sens_func(self, x, softmax_dim=1):
        return F.softmax(self.checker_emb(x), dim=softmax_dim)

    def improve_emb(self, emb, related_concept_index, data_len):
        tar = torch.full((data_len, 1), 0, dtype=torch.float32)
        i = 0
        for con in related_concept_index:
            rest = []
            for c in con:
                if c != 0:
                    rest.append(str(int(c)))
            if len(rest) >= 1:
                cur = "_".join(rest)
                tar[i] = self.question_map[cur]
            i += 1

        return self.question_model(emb, tar.to(self.device))

    def akt_handler(self, q_data, qa_data, target, pid_data=None):
        q_embed_data = self.akt.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.akt.separate_qa:
            # BS, seqlen, d_model #f_(ct,rt)
            qa_embed_data = self.akt.qa_embed(qa_data)
        else:
            qa_data = (qa_data - q_data) // self.akt.n_question  # rt
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.akt.qa_embed(qa_data) + q_embed_data

        q_embed_diff_data = self.akt.q_embed_diff(q_data)  # d_ct
        pid_embed_data = self.akt.difficult_param(pid_data)  # uq
        q_embed_data = q_embed_data + pid_embed_data * \
                       q_embed_diff_data  # uq *d_ct + c_ct
        qa_embed_diff_data = self.akt.qa_embed_diff(
            qa_data)  # f_(ct,rt) or #h_rt
        if self.akt.separate_qa:
            qa_embed_data = qa_embed_data + pid_embed_data * \
                            qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
        else:
            qa_embed_data = qa_embed_data + pid_embed_data * \
                            (qa_embed_diff_data + q_embed_diff_data)  # + uq *(h_rt+d_ct)
        c_reg_loss = (pid_embed_data ** 2.).sum() * self.akt.l2
        self.d_output = self.akt.model(q_embed_data, qa_embed_data)
        concat_q = torch.cat([self.d_output, q_embed_data], dim=-1)
        self.akt_output = self.akt.out(concat_q)
        labels = target.reshape(-1)
        preds = (self.akt_output.reshape(-1))  # logit
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum() + c_reg_loss

    def dkvmn_handler(self, q_data, qa_data, target):
        loss, read_content_embed = self.dkvmn(q_data, qa_data, target)
        self.dkvmn_output = read_content_embed.view(q_data.shape[0], q_data.shape[1], -1)
        return loss