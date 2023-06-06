import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

# def get_noise2(shape, mean, std):
#     return torch.normal(mean, std, size=shape).cuda()

# # this efficient implementation comes from https://github.com/xptree/DeepInf/
# class BatchMultiHeadGraphAttention(nn.Module):  # class GAT中使用(张量的一种高效运算)
#     def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
#         super(BatchMultiHeadGraphAttention, self).__init__()
#         self.n_head = n_head
#         self.f_in = f_in
#         self.f_out = f_out
#         self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
#         self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
#         self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))
#
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(attn_dropout)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(f_out))
#             nn.init.constant_(self.bias, 0)
#         else:
#             self.register_parameter("bias", None)
#
#         nn.init.xavier_uniform_(self.w, gain=1.414)
#         nn.init.xavier_uniform_(self.a_src, gain=1.414)
#         nn.init.xavier_uniform_(self.a_dst, gain=1.414)
#
#     def forward(self, h):
#         bs, n = h.size()[:2]
#         h_prime = torch.matmul(h.unsqueeze(1), self.w)  # 张量相乘
#         attn_src = torch.matmul(h_prime, self.a_src)
#         attn_dst = torch.matmul(h_prime, self.a_dst)
#         attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
#             0, 1, 3, 2
#         )
#         attn = self.leaky_relu(attn)
#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         output = torch.matmul(attn, h_prime)
#         if self.bias is not None:
#             return output + self.bias, attn
#         else:
#             return output, attn
#
#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + " ("
#             + str(self.n_head)
#             + " -> "
#             + str(self.f_in)
#             + " -> "
#             + str(self.f_out)
#             + ")"
#         )


# class GAT(nn.Module):  # 公式4
#     def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
#         super(GAT, self).__init__()
#         self.n_layer = len(n_units) - 1
#         self.dropout = dropout
#         self.layer_stack = nn.ModuleList()
#
#         for i in range(self.n_layer):
#             f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
#             self.layer_stack.append(
#                 BatchMultiHeadGraphAttention(
#                     n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
#                 )
#             )
#
#         self.norm_list = [
#             torch.nn.InstanceNorm1d(32).cuda(),
#             torch.nn.InstanceNorm1d(64).cuda(),
#         ]
#
#     def forward(self, x):
#         bs, n = x.size()[:2]
#         for i, gat_layer in enumerate(self.layer_stack):
#             x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
#             x, attn = gat_layer(x)
#             if i + 1 == self.n_layer:
#                 x = x.squeeze(dim=1)
#             else:
#                 x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
#                 x = F.dropout(x, self.dropout, training=self.training)
#         else:
#             return x


# class GATEncoder(nn.Module): # G-LSTM编码 + M-LSTM和G-LSTM结果拼接为h
#     def __init__(self, n_units, n_heads, dropout, alpha):
#         super(GATEncoder, self).__init__()
#         self.gat_net = GAT(n_units, n_heads, dropout, alpha)
#
#     def forward(self, obs_traj_embedding, seq_start_end):
#         graph_embeded_data = []
#         for start, end in seq_start_end.data:
#             curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]  # M-LSTM
#             curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)  # G-LSTM
#             graph_embeded_data.append(curr_seq_graph_embedding)  # 拼接为h
#         graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
#         return graph_embeded_data


"""
f_gcn = [5,16,32,64]
f_atten = [16,16,16]
"""
class EGCN(nn.Module):  # 传播过滤器
    def __init__(self, f_in, f_gcn, f_atten, channels=2):
        super(EGCN, self).__init__()
        self.f_in = f_in
        self.f_gcn = f_gcn
        self.f_atten = f_atten
        self.channels = channels  # 边缘特征的索引是1--4：没有方向角

        self.w_atten = nn.Parameter(torch.Tensor(self.channels, self.f_atten))
        # nn.Parameter(torch.Tensor()):将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.bn = nn.BatchNorm1d(self.channels)  # bn
        self.fc = nn.Linear(self.f_in, self.f_gcn)  # fc
        self.bn2 = nn.BatchNorm1d(self.f_gcn)

        # initialize custom parameters
        nn.init.xavier_uniform_(self.w_atten, gain=1.414)

    def get_adj(self, bs, E_): # 得到邻接矩阵
        A = []
        for i in range(bs): # bs代表人的数量
            A.append(torch.matmul(torch.matmul(torch.matmul(E_[i],
                                                            self.w_atten), self.w_atten.t()), E_[i].t()))
        A = torch.sum(torch.stack(A, 0), 0) # 公式6 [bs,bs]:主要目的是把上面[tensor[],tensor[]..]变成tensor[[],[]]
        A_ = F.softmax(A, 1).unsqueeze(2).repeat(1, 1, self.channels) # 公式7，[bs,bs,channels]
        '''
        torch.nn.functional.softmax(input, dim):
        返回结果是一个与输入维度相同的张量，每个元素的取值范围在（0,1）区间。
        参数：dim:指明维度，dim=0表示按列计算；dim=1表示按行计算。
        '''
        return A_

    def get_h(self, A_adj, x):  # 公式8
        # A_adj:[n,n,8];x:[n,8]
        H = []
        for k in range(self.channels):
            H.append(self.fc(torch.matmul(A_adj[:, :, k], x))) # 公式8括号部分，H0=X

        H = torch.sum(torch.stack(H, 0), 0)
        H = self.bn2(H)  # 归一化
        H = torch.tanh(H)
        return H

    def forward(self, x, E):
        # # E：[bs,bs,4]  相对状态(边矩阵)
        # bs = x.shape[0] # x:[bs,5];bs为人的数量
        # E_ = E.permute(0, 2, 1)  # [bs,4,bs]
        # E_ = self.bn(E_).permute(0, 2, 1)  # bn:归一化；E_:[bs,bs,4]
        #
        # A_ = self.get_adj(bs, E_)  # A_:[bs,bs,4]
        # A_adj = E_ * A_  # A_adj:[bs,bs,4]
        # H = self.get_h(A_adj, x)  # [bs,f_gcn]
        bs = x.shape[0]  # bs为人的数量  # E：[n,n,8] ; x:[n,8]
        E_ = E.permute(0, 2, 1)  # [n,8,n]
        E_ = self.bn(E_).permute(0, 2, 1)  # bn:归一化；E_:[n,n,8]

        A_ = self.get_adj(bs, E_)  # A_:[n,n,8]
        A_adj = E_ * A_  # A_adj:[n,n,8]
        H = self.get_h(A_adj, x)  # [n,f_gcn]
        # print("H.shape:",H.shape)  # [n,16]
        del E_,A_,A_adj
        return H


class EGCNBLOCK(nn.Module):
    def __init__(self, f_gcn, f_atten):
        super(EGCNBLOCK, self).__init__()
        self.n_layer = len(f_gcn) - 1  # layers size：EGCN有三层
        # flgcn代表第l层的EGCN层有多少个传播过滤器，文章中第0层是5
        self.layer_stack = nn.ModuleList()
        # ModuleList：它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器(可以当做list用)

        for i in range(self.n_layer):
            self.layer_stack.append(   # 将每个EGCN层按顺序存储到layer_stack中
                EGCN(f_gcn[i], f_gcn[i + 1], f_atten[i])
            )

    def forward(self, x, E):
        # x:[8,n,32],E：[n,n,8,32]
        # x = x.permute(1, 0, 2)
        # x = self.layer1(x)  # [n,8,1]
        # x = x.squeeze(dim=2)  # [n,8]
        # # print("x.shape:",x.shape)
        #
        # E = self.layer1(E)  # [n,n,8,1]
        # E = E.squeeze(dim=3)  # [n,n,8]
        # # print("E.shape:", E.shape)

        for _, egcn_layer in enumerate(self.layer_stack): # 遍历layer_stack容器
            torch.cuda.empty_cache()  # nn.Linear(m, n)使用O(nm)内存
            x = egcn_layer(x, E)

        return x

def angle_matrix(input_matrix):  # 传入相对位置矩阵
    cos_array = np.zeros((input_matrix.shape[0], input_matrix.shape[1], input_matrix.shape[2]))
    # 有两个点point(x1, y1)和point(x2, y2);
    # 两个点形成的斜率的角度计算方法分别是：
    # float angle = atan2(y2 - y1, x2 - x1);
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            x = input_matrix[i][j][0]
            y = input_matrix[i][j][1]
            angle = math.atan2(y, x)  # 由x,y计算反正切值
            angle = int(angle * 180 / math.pi)  # 由反正切值计算角度
            radian = math.radians(angle)  # 角度转弧度
            cosvalue = math.cos(radian)  # 弧度转余弦值
            cos_array[i][j][0] = cosvalue  # 将余弦值写入角度矩阵
    return cos_array

class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,
        traj_lstm_hidden_size,
        graph_network_out_dims,
        graph_lstm_hidden_size,
        noise_dim=(8,),
        noise_type="gaussian",
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        # self.gatencoder = GATEncoder(
        #     n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        # )
        self.egcnblock = EGCNBLOCK(f_gcn=[2, 8, 32], f_atten=[8, 8])

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = (self.traj_lstm_hidden_size + self.graph_lstm_hidden_size + noise_dim[0])# 32+32+16

        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size) # 2,32
        self.graph_lstm_model = nn.LSTMCell(graph_network_out_dims, graph_lstm_hidden_size)  # 32,32

        self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size, 2)  # 32,2
        self.traj_gat_hidden2pos = nn.Linear(self.traj_lstm_hidden_size + self.graph_lstm_hidden_size, 2)  # 64,2
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, 2)   # 32+32+16, 2

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, self.pred_lstm_hidden_size)  # 2, 32+32+16

    def init_hidden_traj_lstm(self, batch):  # 轨迹编码的lstm的hidden初始化
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch): # GAT编码之后的lstm的hidden初始化
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        # [n,64][64,2]
        noise_shape = (seq_start_end.size(0),) + self.noise_dim  # noise_dim:16
        z_decoder = get_noise(noise_shape, self.noise_type)  # noise shape:[batch_size,16]?
        # z_decoder = get_noise2(noise_shape, mean=mean, std=std)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def get_edge(self, x):
        bs = x.shape[0]  # [n,2]
        channels = x.shape[1] - 1
        x_ = x[:, :]  # [bs,4]

        # 相对矩阵E计算
        x_m = x_.unsqueeze(1).repeat(1, bs, 1)  # [bs,bs,4]:各维度重复1次，bs次，1次
        x_l = x_.unsqueeze(0).repeat(bs, 1, 1)  # [bs,bs,4]
        E = torch.abs(x_m - x_l)

        # E_D = torch.sum(E, 1)  # [bs,4];将第一个维度求和
        # for i in range(bs):
        #     for k in range(channels):
        #         E[i][i][k] = E_D[i][k] if E_D[i][k] > self.thresold else 1

        # bs = x.shape[1] # [8,n,32]
        # x_ = x.permute(1, 0, 2) # [n,8,32]
        # # x_ = x_[:, :-1]
        #
        # x_m = x_.unsqueeze(1).repeat(1, bs, 1, 1)  # [n,n,8,32]
        # # torch.cuda.empty_cache()
        # x_l = x_.unsqueeze(0).repeat(bs, 1, 1, 1)  # [n,n,8,32]
        #
        # E = torch.abs(x_m - x_l)

        return E

    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        seq_start_end,  # [64,2]
        teacher_forcing_ratio=0.5,
        training_step=3,
    ):
        batch = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batch)
        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []

        # # 计算观测轨迹均值和方差
        # obs_traj_pos_numpy = obs_traj_pos.cpu().numpy()
        # mean = np.mean(obs_traj_pos_numpy)
        # std = np.std(obs_traj_pos_numpy)

        # print("obs", obs_traj_rel.shape)  # [20,4 2]

        # 角度编码
        angle_embedding_ndarray = angle_matrix(obs_traj_rel)  # [8,n,2]
        angle_embedding_tensor = torch.from_numpy(angle_embedding_ndarray).float().cuda()

        obs_traj_rel = obs_traj_rel + angle_embedding_tensor

        # 0-150次迭代training_step=1；150-250次training_step=2；250次以上training_step=3
        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(
                obs_traj_rel[: self.obs_len].size(0), dim=0
            )
        ):
            # print("traj_lstm_h:", traj_lstm_h_t.shape)
            # print("traj_lstm_c_t:",traj_lstm_c_t.shape)
            # print("input_t.shape:",input_t.shape) [1,4,2]
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(   # e-lstm编码
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            # print("traj_lstm_h_t.shape:", traj_lstm_h_t.shape)
            if training_step == 1:  # train e-lstm
                output = self.traj_hidden2pos(traj_lstm_h_t) # traj_hidden2pos:nn.linear
                # print("output.shape:", output.shape)
                pred_traj_rel += [output]
            else:
                traj_lstm_hidden_states += [traj_lstm_h_t]

        if training_step == 2: # train e-lstm,gat,g-lstm
            # traj_lstm_hidden_states:[8, ni, 32]; seq_start_end: [64,2]
            # print("torch.stack:", torch.stack(traj_lstm_hidden_states).shape)
            # graph_lstm_input = self.gatencoder(
            #     torch.stack(traj_lstm_hidden_states), seq_start_end  # torch.stack数组变张量
            # )    # 改动处
            traj_lstm_hidden_states = torch.stack(traj_lstm_hidden_states)
            x = obs_traj_rel  # [8,n,2]
            graph_lstm_input = []
            for i, x_input in enumerate(
                    x[: self.obs_len].chunk(
                        x[: self.obs_len].size(0), dim=0
                    )
            ):
                E = self.get_edge(x_input.squeeze(0))  # [n,n,2]
                egcn_output = self.egcnblock(x_input.squeeze(0), E)  # [n,2],[n,n,2]-->[n,32]
                graph_lstm_input += [egcn_output]
            graph_lstm_input = torch.stack(graph_lstm_input)
            # print("graph_lstm_input:",graph_lstm_input.shape) # [8,n,32]
            # graph_lstm_input = graph_lstm_input.unsqueeze(0).repeat(8, 1, 1) # [8,n,32]
            for i in range(self.obs_len):
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    graph_lstm_input[i], (graph_lstm_h_t, graph_lstm_c_t)
                )
                encoded_before_noise_hidden = torch.cat(
                    (traj_lstm_hidden_states[i], graph_lstm_h_t), dim=1
                )
                # print("encode_before_noise_hidden:", encoded_before_noise_hidden.shape)
                output = self.traj_gat_hidden2pos(encoded_before_noise_hidden)
                # print("output:", output.shape)
                pred_traj_rel += [output]
        if training_step == 3:  # train e-lstm,gat,g-lstm,d-lstm
            # graph_lstm_input = self.gatencoder(
            #     torch.stack(traj_lstm_hidden_states), seq_start_end
            # )
            # x = torch.stack(traj_lstm_hidden_states)
            # E = self.get_edge(x)
            # graph_lstm_input = self.egcnblock(x, E)
            # graph_lstm_input = graph_lstm_input.unsqueeze(0).repeat(8, 1, 1)
            traj_lstm_hidden_states = torch.stack(traj_lstm_hidden_states)
            x = obs_traj_rel
            graph_lstm_input = []
            for i, x_input in enumerate(
                    x[: self.obs_len].chunk(
                        x[: self.obs_len].size(0), dim=0
                    )
            ):
                E = self.get_edge(x_input.squeeze(0))  # [n,n,2]
                egcn_output = self.egcnblock(x_input.squeeze(0), E)
                # print("egcn_output:", egcn_output.shape)
                graph_lstm_input += [egcn_output]
            graph_lstm_input = torch.stack(graph_lstm_input)
            for i, input_t in enumerate(
                graph_lstm_input[: self.obs_len].chunk(graph_lstm_input[: self.obs_len].size(0), dim=0)
                # chunk(a,b),a表示分成的块数，b=0沿横向分割，b=1沿纵向分割
            ):
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    input_t.squeeze(0), (graph_lstm_h_t, graph_lstm_c_t)
                )
                graph_lstm_hidden_states += [graph_lstm_h_t]

        if training_step == 1 or training_step == 2:
            return torch.stack(pred_traj_rel)
            # step=1时pred_traj_rel为e-lstm输出；
            # step=2时pred_traj_rel为e-lstm和g-lstm输出拼接加全连接
        else:
            # 拼接两个结果
            encoded_before_noise_hidden = torch.cat((traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]), dim=1)
            # print("encode_before_noise_hidden:",encoded_before_noise_hidden.shape)
            pred_lstm_hidden = self.add_noise(encoded_before_noise_hidden, seq_start_end)# 添加噪声[n,64][64,2]
            # pred_lstm_hidden = self.add_noise(encoded_before_noise_hidden, seq_start_end, mean, std)
            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
            output = obs_traj_rel[self.obs_len-1]
            if self.training:  # 如果在训练阶段，train.py是true,evaluate时为false
                for i, input_t in enumerate(
                    obs_traj_rel[-self.pred_len:].chunk(obs_traj_rel[-self.pred_len:].size(0), dim=0)
                ):  # .chunk-->沿轴0分成obs_traj_rel[-self.pred_len:].size(0),即8块
                    # input_t有一定概率不变，有一定概率是output.unsqueeze(0)
                    teacher_force = random.random() < teacher_forcing_ratio # teacher_forcing_ratio=0.5
                    input_t = input_t if teacher_force else output.unsqueeze(0)

                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(  # 预测部分的d-lstm
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            else:
                for i in range(self.pred_len):
                    # 测试时输入的是历史相对轨迹--》390行
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            return outputs
