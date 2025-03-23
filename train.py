import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging as log
import numpy
import tqdm
import pickle
from utils import batch_data_to_device

# train 函数接受三个参数：model（深度学习模型），loaders（数据加载器的字典），和 args（包含各种训练参数的配置）
def train(model, loaders, args):
    log.info("training...")
    # 创建一个二元交叉熵损失函数（BCELoss），用于度量模型输出与真实标签之间的差异
    BCELoss = torch.nn.BCEWithLogitsLoss()
    # 创建一个 Adam 优化器（optimizer），用于更新模型的参数。学习率为 args.lr，权重衰减为 args.decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # 创建 Sigmoid 函数（train_sigmoid），用于将模型的输出映射到范围 [0, 1] 内
    train_sigmoid = torch.nn.Sigmoid()
    # 初始化 show_loss 为 100，用于存储训练损失的显示值
    show_loss = 100
    for epoch in range(args.n_epochs):
        loss_all = 0
        # 遍历 loaders['train'] 数据加载器中的每个数据批次，其中 data 包含输入数据和标签
        for step, data in enumerate(loaders['train']):  # data[0]=(1,11),data[1]=(1,626)
            # with torch.no_grad() 上下文管理器，禁用梯度计算，因为这一部分是前向传播而不需要梯度信息
            with torch.no_grad():
                # 将数据移动到计算设备（GPU 或 CPU）上，同时设置模型为训练模式：model.train()
                x, y = batch_data_to_device(data, args.device)
            model.train()
            # 获取数据长度 data_len，然后初始化隐藏状态 h 为零，并准备用于存储其他信息的各种列表和张量
            data_len = len(x[0])
            h = torch.zeros(data_len, args.dim).to(args.device)  # h=(11,64)
            p_action_list, pre_state_list, emb_action_list, op_action_list, actual_label_list, states_list, reward_list, predict_list, ground_truth_list = [], [], [], [], [], [], [], [], []
            '''p_action_list：存储问题认知部分的动作编号，每个动作编号对应一个概念编码。

•  pre_state_list：存储问题的先前状态，每个先前状态是由问题特征和隐藏状态拼接而成的向量。

•  emb_action_list：存储问题敏感部分的动作编号，每个动作编号对应一个概念编码。

•  op_action_list：存储问题操作的预测标签，每个预测标签是一个二进制值，表示模型预测学生是否回答正确。

•  actual_label_list：存储问题操作的真实标签，每个真实标签也是一个二进制值，表示学生实际是否回答正确。

•  states_list：存储问题的当前状态，每个当前状态是由问题的真实表示和预测表示连接而成的向量。

•  reward_list：存储问题的奖励值，每个奖励值是一个标量，表示模型预测是否与真实标签一致。

•  predict_list：存储问题的预测概率值，每个预测概率值是一个标量，表示模型输出学生回答正确的概率。

•  ground_truth_list：存储问题操作的真实标签，与actual_label_list相同。
这些变量都是在时间步循环中不断更新和添加的，最后在循环结束后合并成张量，以便进行批处理和损失计算'''
            # 初始化 rt_x 为一个全零张量
            rt_x = torch.zeros(data_len, 1, args.dim * 2).to(args.device)  # (11,1,128)
            q_data = torch.zeros((data_len, 200), dtype=torch.int64)
            qa_data = torch.zeros((data_len, 200), dtype=torch.int64)
            pid_data = torch.zeros((data_len, 200), dtype=torch.int64)
            tar = torch.full((data_len, 200), -1, dtype=torch.float32)
            first = True
            for i in range(0, args.seq_len):
                pid_bs = x[1][i][6]
                for j in range(0, len(pid_bs)):
                    pid_data[j][i] = pid_bs[j]
                cons = x[1][i][8]
                target = x[1][i][4]
                for j in range(0, len(cons)):
                    q_data[j][i] = cons[j].item()
                    if cons[j].item() != 0:
                        qa_data[j][i] = cons[j].item() + 143 * target[j].item()
                        tar[j][i] = target[j]

            if first:
                first = False

            loss_akt = model.akt_handler(q_data.to(args.device), qa_data.to(args.device),
                                         tar.to(args.device), pid_data.to(args.device))
            # loss_dkvmn = model.dkvmn_handler(q_data.to(args.device), qa_data.to(args.device),
            #                                  tar.to(args.device))
            # 循环遍历每个时间步（seqi）
            for seqi in range(0, args.seq_len):  # arg.seq_len=200
                # 创建问题表示 ques_h，(就是论文里面的hv变量)通过将问题特征、隐藏状态 h 连接在一起而得到
                ques_h = torch.cat([
                    model.get_ques_representation(x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][5].size()[0]),
                    h], dim=1)  # (11,192)
                # 使用模型中的 pi_cog_func 函数构建问题的认知概率分布 flip_prob_emb
                flip_prob_emb = model.pi_cog_func(ques_h)  # (11,10)
                # 通过 Categorical 创建一个分类分布 m，然后从中抽样出一个动作 emb_ap，并获取对应的概念编码 emb_p
                m = Categorical(flip_prob_emb)
                emb_ap = m.sample()
                emb_p = model.cog_matrix[emb_ap, :]  # (11,128)
                # 使用模型的 obtain_v 函数，根据问题的输入特征、隐藏状态 h，以及 emb_p 更新隐藏状态 h 并得到模型输出的概率值 prob
                h_v, v, logits, rt_x = model.obtain_v(x[1][seqi], h, rt_x, emb_p,
                                                      seqi)  # h_v(11,192) logits(11,1) rt_x(11,1,128)
                '''  h_v：这个变量表示问题的隐藏状态，它是由模型的gru层根据问题的输入特征、隐藏状态h和概念编码emb_p计算得到的。它可以反映学生对问题的理解程度和知识点的掌握情况。

•  v：这个变量表示问题的价值函数，它是由模型的v_func层根据问题的隐藏状态h_v计算得到的。它可以反映学生回答问题的期望奖励，也就是模型预测学生回答正确的概率。

•  logits：这个变量表示问题的输出对数几率，它是由模型的logits_func层根据问题的隐藏状态h_v计算得到的。它可以通过Sigmoid函数转换成问题的输出概率值prob，也就是模型预测学生回答正确的概率。

•  rt_x：这个变量表示问题的记忆状态，它是由模型的rt_func层根据问题的输入特征、隐藏状态h和概念编码emb_p计算得到的。它可以反映学生对问题的记忆效果和注意力分配'''
                # 计算学生正确回答问题qt的概率(prob就是论文里CE模块的即rt^)
                prob = train_sigmoid(logits)  # (11,1)
                # 根据问题操作的真实标签 x[1][seqi][4]也就是学生回答问题的结果构建两个张量，分别为 out_operate_groundtruth （是一个二进制向量，表示学生是否回答正确）和
                # out_x_groundtruth（拼接了隐藏状态h和问题操作的真实标签的向量，表示问题的真实状态）
                out_operate_groundtruth = x[1][seqi][4]  # (11,1)
                out_x_groundtruth = torch.cat([
                    h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                    h_v.mul((1 - out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                    dim=1)  # (11,384)
                # 基于模型输出的概率值 prob，使用阈值 0.5 得到问题操作的二进制预测 out_operate_logits（一个二进制向量，表示模型预测学生是否回答正确）,并构建问题的二进制表示
                # out_x_logits（拼接了隐藏状态h和问题操作的预测标签的向量，表示问题的预测状态）
                out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(args.device),
                                                 torch.tensor(0).to(args.device))  # (11,1)
                out_x_logits = torch.cat([
                    h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                    h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                    dim=1)  # (11,384)
                # 将问题的真实表示和预测表示连接在一起，得到问题的最终表示 out_x (Vm) （这个向量包含了问题的真实和预测信息，用于计算感知选择概率和奖励）
                out_x = torch.cat([out_x_groundtruth, out_x_logits], dim=1)
                # 获取问题操作的真实标签
                ground_truth = x[1][seqi][4].squeeze(-1)
                # 使用模型的 pi_sens_func 函数构建问题的敏感性概率分布 flip_prob_emb
                flip_prob_emb = model.pi_sens_func(out_x)
                # 类似认知部分，从中抽样一个动作 emb_a 并获取对应的概念编码 emb
                m = Categorical(flip_prob_emb)
                emb_a = m.sample()
                emb = model.acq_matrix[emb_a, :]
                # 更新隐藏状态 h，同时计算奖励（reward）
                h = model.update_state(h, v, emb, ground_truth.unsqueeze(1))
               
                emb_action_list.append(emb_a)
                p_action_list.append(emb_ap)
                states_list.append(out_x)
                pre_state_list.append(ques_h)

                ground_truth_list.append(ground_truth)
                predict_list.append(logits.squeeze(1))
                this_reward = torch.where(out_operate_logits.squeeze(1).float() == ground_truth,
                                          torch.tensor(1).to(args.device),
                                          torch.tensor(0).to(args.device))
                reward_list.append(this_reward)
            # 在时间步循环结束后，将各种信息合并成张量，以便进行批处理
            seq_num = x[0]
            emb_action_tensor = torch.stack(emb_action_list, dim=1)
            p_action_tensor = torch.stack(p_action_list, dim=1)
            state_tensor = torch.stack(states_list, dim=1)
            pre_state_tensor = torch.stack(pre_state_list, dim=1)
            reward_tensor = torch.stack(reward_list, dim=1).float() / (
                seq_num.unsqueeze(-1).repeat(1, args.seq_len)).float()
            logits_tensor = torch.stack(predict_list, dim=1)
            ground_truth_tensor = torch.stack(ground_truth_list, dim=1)
            # 创建空列表 loss用于存储损失值
            loss = []
            # 创建两个空列表 tracat_logits 和 tracat_ground_truth用于存储模型的输出和真实标签用于计算二元交叉熵损失
            tracat_logits = []
            tracat_ground_truth = []
            # 开始循环，迭代 data_len 次，其中 data_len 表示数据的样本数
            for i in range(0, data_len):
                # 获取当前样本的序列长度 this_seq_len 和对应的奖励列表 this_reward_list
                this_seq_len = seq_num[i]  # seq_num=(1,11)
                this_reward_list = reward_tensor[i]
                # 创建 this_cog_state构建问题的认知状态
                # 将先前状态（pre_state_tensor[i]）的前 this_seq_len 个元素与全零张量连接起来用于构建问题的认知状态
                this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                            torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(args.device)
                                            ], dim=0)
                # 创建 this_sens_state构建问题的敏感状态，将当前状态（state_tensor[i]）的前 this_seq_len 个元素与全零张量连接起来，用于构建问题的敏感状态
                this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                             torch.zeros(1, state_tensor[i][0].size()[0]).to(args.device)
                                             ], dim=0)
                # 创建 td_target_cog，它是问题认知部分的奖励列表的前 this_seq_len 个元素，将其形状变为列向量
                # [0: this_seq_len]选择 this_reward_list 中的前 this_seq_len 个元素
                # .unsqueeze(1): 将选定的奖励值列表从一维张量变换为二维张量，维度 (N,) 变成了 (N, 1)
                # 在深度学习中，通常用二维张量来表示目标或标签数据。
                td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
                # 创建 delta_cog，它等于 td_target_cog 的一个副本。
                # 同时，将 delta_cog 转移到 CPU 并转化为 NumPy 数组。
                '''.detach()是PyTorch张量方法，用于创建一个新的张量，其计算图不依赖于原始张量（delta_cog）。
                这意味着将新张量与原张量分离，不再跟踪其计算历史。通常在计算损失函数时，将目标数据分离，以防止梯度传播到目标数据'''
                delta_cog = td_target_cog
                delta_cog = delta_cog.detach().cpu().numpy()
                # 创建 td_target_sens，它是问题敏感部分（KASE）的奖励列表的前 this_seq_len 个元素，将其形状变为列向量
                td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
                delta_sens = td_target_sens
                delta_sens = delta_sens.detach().cpu().numpy()
                # 创建一个空列表 advantage_lst_cog 和初始化 advantage 为 0.0。
                # 接下来，计算问题认知部分的优势（advantage）值
                advantage_lst_cog = []
                advantage = 0.0
                # 遍历 delta_cog 列表，计算每个时间步上的优势值，然后将它们添加到 advantage_lst_cog 列表中。
                # 最后，反转列表中的元素，以匹配时间步的顺序
                for delta_t in delta_cog[::-1]:
                    advantage = args.gamma * advantage + delta_t[0]
                    advantage_lst_cog.append([advantage])
                advantage_lst_cog.reverse()
                # 创建 advantage_cog，将 advantage_lst_cog 转化为 PyTorch 张量，并移动到计算设备（GPU 或 CPU）上
                advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(args.device)
                # 使用模型的 pi_cog_func 函数计算问题认知部分的动作概率分布 pi_cog
                # 并从中取出相应的动作概率 pi_a_cog
                pi_cog = model.pi_cog_func(this_cog_state[:-1])
                pi_a_cog = pi_cog.gather(1, p_action_tensor[i][0: this_seq_len].unsqueeze(1))
                #  计算问题认知部分的损失 loss_cog，它是负对数似然乘以优势
                loss_cog = -torch.log(pi_a_cog) * advantage_cog
                # 将问题认知部分的损失 loss_cog 的总和添加到 loss 列表中。
                loss.append(torch.sum(loss_cog))
                # 创建一个空列表 advantage_lst_sens 和初始化 advantage 为 0.0。接下来，计算问题敏感部分的优势（advantage）值
                advantage_lst_sens = []
                advantage = 0.0
                # 遍历 delta_sens 列表，计算每个时间步上的优势值，然后将它们添加到 advantage_lst_sens 列表中
                # 最后，反转列表中的元素，以匹配时间步的顺序
                for delta_t in delta_sens[::-1]:
                    # advantage = args.gamma * args.beta * advantage + delta_t[0]
                    advantage = args.gamma * advantage + delta_t[0]
                    advantage_lst_sens.append([advantage])
                advantage_lst_sens.reverse()
                # 创建 advantage_sens，将 advantage_lst_sens 转化为 PyTorch 张量，并移动到计算设备上
                advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(args.device)
                # 使用模型的 pi_sens_func 函数计算问题敏感部分的动作概率分布 pi_sens，并从中取出相应的动作概率 pi_a_sens
                pi_sens = model.pi_sens_func(this_sens_state[:-1])
                pi_a_sens = pi_sens.gather(1, emb_action_tensor[i][0: this_seq_len].unsqueeze(1))
                # 计算问题敏感部分的损失 loss_sens，它同样是负对数似然乘以优势
                loss_sens = - torch.log(pi_a_sens) * advantage_sens
                # 将问题敏感部分的损失 loss_sens 的总和添加到 loss 列表中
                loss.append(torch.sum(loss_sens))

                # 获取问题认知部分和敏感部分的预测概率 this_prob，以及对应的真实标签 this_groud_truth
                this_prob = logits_tensor[i][0: this_seq_len]
                this_groud_truth = ground_truth_tensor[i][0: this_seq_len]
                # 将 this_prob 添加到 tracat_logits 列表中，将 this_groud_truth 添加到 tracat_ground_truth 列表中
                tracat_logits.append(this_prob)
                tracat_ground_truth.append(this_groud_truth)

            loss.append(loss_akt)
            # loss.append(loss_dkvmn)
            # 循环结束后，将问题认知部分和敏感部分的损失值合并到一个总的损失 loss 中
            bce = BCELoss(torch.cat(tracat_logits, dim=0), torch.cat(tracat_ground_truth, dim=0))
            # 获取标签的总长度 label_len，并计算总的损失值 loss_l，将 loss 列表中的各项损失值求和
            label_len = torch.cat(tracat_ground_truth, dim=0).size()[0]
            loss_l = sum(loss)
            # 计算最终的损失，它是权重项 args.lamb 乘以平均损失值，再加上二元交叉熵损失 bce
            loss = args.lamb * (loss_l / label_len) + bce
            # 累积总的损失 loss_all
            loss_all += loss
            # 执行梯度清零操作，即 optimizer.zero_grad()，以准备进行反向传播
            optimizer.zero_grad()
            # 计算总的损失 loss 的梯度
            loss.backward()
            # 使用优化器 optimizer 执行参数更新
            optimizer.step()
        # 计算并打印当前轮次的平均损失值 show_loss
        show_loss = loss_all / len(loaders['train'].dataset)
        model.draw()
        model.clear()
        # 使用 evaluate 函数计算模型在验证数据集上的性能，包括准确率 acc 和 AUC 值 auc
        acc, auc = evaluate(model, loaders['valid'], args)
        # 同样，使用 evaluate 函数计算模型在测试数据集上的性能，包括测试准确率 tacc 和测试 AUC 值 tauc
        tacc, tauc = evaluate(model, loaders['test'], args)
        # 使用 log.info 打印当前轮次的训练信息，包括轮次、损失值和性能指标
        log.info(
            'Epoch: {:03d}, Loss: {:.7f}, valid acc: {:.7f}, valid auc: {:.7f}, test acc: {:.7f}, test auc: {:.7f}'.format(
                epoch, show_loss, acc, auc, tacc, tauc))
        # 如果定义了 args.save_every 并且当前轮次是保存周期的倍数，将模型参数保存到文件
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model, os.path.join(args.run_dir, 'params_%i.pt' % epoch))


def evaluate(model, loader, args):
    model.eval()
    eval_sigmoid = torch.nn.Sigmoid()
    y_list, prob_list, final_action = [], [], []

    for step, data in enumerate(loader):

        with torch.no_grad():
            x, y = batch_data_to_device(data, args.device)
        model.train()
        data_len = len(x[0])
        h = torch.zeros(data_len, args.dim).to(args.device)
        batch_probs, uni_prob_list, actual_label_list, states_list, reward_list = [], [], [], [], []
        H = None
        if 'eernna' in args.model:
            H = torch.zeros(data_len, 1, args.dim).to(args.device)
        else:
            H = torch.zeros(data_len, args.concept_num - 1, args.dim).to(args.device)
        rt_x = torch.zeros(data_len, 1, args.dim * 2).to(args.device)
        q_data = torch.zeros((data_len, 200), dtype=torch.int64)
        qa_data = torch.zeros((data_len, 200), dtype=torch.int64)
        pid_data = torch.zeros((data_len, 200), dtype=torch.int64)
        tar = torch.full((data_len, 200), -1, dtype=torch.float32)
        for i in range(0, args.seq_len):
            pid_bs = x[1][i][6]
            for j in range(0, len(pid_bs)):
                pid_data[j][i] = pid_bs[j]
            cons = x[1][i][8]
            target = x[1][i][4]
            for j in range(0, len(cons)):
                q_data[j][i] = cons[j].item()
                if cons[j].item() != 0:
                    qa_data[j][i] = cons[j].item() + 143 * target[j].item()
                    tar[j][i] = target[j]

        model.akt_handler(q_data.to(args.device), qa_data.to(args.device),
                          tar.to(args.device), pid_data.to(args.device))
        # model.dkvmn_handler(q_data.to(args.device), qa_data.to(args.device),
        #                     tar.to(args.device))
        for seqi in range(0, args.seq_len):
            ques_h = torch.cat([
                model.get_ques_representation(x[1][seqi][6], x[1][seqi][2], x[1][seqi][5], x[1][seqi][5].size()[0]),
                h], dim=1)
            flip_prob_emb = model.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = model.cog_matrix[emb_ap, :]

            h_v, v, logits, rt_x = model.obtain_v(x[1][seqi], h, rt_x, emb_p, seqi)
            prob = eval_sigmoid(logits)
            out_operate_groundtruth = x[1][seqi][4]
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                dim=1)

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(args.device),
                                             torch.tensor(0).to(args.device))
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim=1)
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim=1)

            ground_truth = x[1][seqi][4].squeeze(-1)

            flip_prob_emb = model.pi_sens_func(out_x)

            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = model.acq_matrix[emb_a, :]

            h = model.update_state(h, v, emb, ground_truth.unsqueeze(1))
            uni_prob_list.append(prob.detach())

        seq_num = x[0]
        prob_tensor = torch.cat(uni_prob_list, dim=1)
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            batch_probs.append(prob_tensor[i][0: this_seq_len])
        batch_t = torch.cat(batch_probs, dim=0)
        prob_list.append(batch_t)
        y_list.append(y)

    y_tensor = torch.cat(y_list, dim=0).int()
    hat_y_prob_tensor = torch.cat(prob_list, dim=0)

    acc = accuracy_score(y_tensor.cpu().numpy(), (hat_y_prob_tensor > 0.5).int().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(y_tensor.cpu().numpy(), hat_y_prob_tensor.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return acc, auc
