import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
import pandas as pd

cuda = True if torch.cuda.is_available() else False


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    # 计算样本权重的函数，如果参数use_sample_weight设置为False，那么所有类别的权重相等
    # 否则权重将根据每个类别样本数量的比例而定
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    # 更详细地说，首先进行计数，以查明每个类别有多少个样本。
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    # 然后，每个样本的权重将是其类别的样本总数占所有样本数量的比例。
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):
    # 将一个一维的类别标签张量转换为one-hot表示的二维张量
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    # 根据余弦定理计算两个张量之间的余弦距离
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    # 用点积除以它们的二范数来计算
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    # 将一个密集的张量转换为稀疏张量
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        # 如果x全为零，则返回零值稀疏张量
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    # 获取x中非零值的位置和值
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    # 计算生成邻接矩阵的参数，目前只实现了余弦距离的情况
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1, )).values[edge_per_node * data.shape[0]]
    return np.asscalar(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    # 根据距离矩阵和参数生成图，对角线上的元素将被设置为0
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    # 生成邻接矩阵的函数，目前只实现了余弦距离的情况
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    # 根据/graph_from_dist_tensor的结果g，计算邻接矩阵
    # 这里邻接矩阵的值为1-余弦距离，即余弦相似度
    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError
    # 通过元素乘法保留g中的非零值
    adj = adj * g
    # 确保邻接矩阵是对称的，取最大值，最后认为邻接矩阵是对称的
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    # 最后，对每行的邻接矩阵进行归一化，使其成为转移矩阵，并转换为稀疏阵表示
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    # 生成测试邻接矩阵的函数，输入数据，训练和测试索引，参数和度量（目前只实现余弦）
    assert metric == "cosine", "Only cosine distance implemented"
    # 初始化零矩阵
    adj = torch.zeros((data.shape[0], data.shape[0]))
    # 如果使用cuda，则将adj放置到GPU上
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])

    # 计算训练集和测试集之间的距离
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)

    # 生成训练集到测试集的邻接矩阵
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    # 对邻接矩阵进行mask
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    # 计算测试集和训练集之间的距离
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)

    # 生成测试集到训练集的邻接矩阵
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    # 再次进行mask
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr  # retain selected edges

    # 使邻接矩阵保持对称
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    # 对邻接矩阵进行归一化
    adj = F.normalize(adj + I, p=1)
    # 最后，将邻接矩阵转换成稀疏格式
    adj = to_sparse(adj)

    return adj


def save_model_dict(folder, model_dict):
    # 保存模型参数的函数，为每个模块在指定的目录下创建一个文件
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module + ".pth"))


def load_model_dict(folder, model_dict):
    # 加载模型的参数，从指定的目录中读取每个模块的文件
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module + ".pth")):
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module + ".pth"),
                                                          map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        # 如果使用cuda，则将模型放置到GPU上
        if cuda:
            model_dict[module].cuda()
    return model_dict

def cal_normalized_laplacian(adj_matrix):
    """
    计算对称归一化的图拉普拉斯矩阵
    """
    if not adj_matrix.is_sparse:
        raise ValueError("Input matrix should be a sparse tensor.")

    # 转换为CPU上的密集张量以进行操作
    adj_matrix_dense = adj_matrix.to_dense().cpu()

    # 计算度矩阵
    degrees = adj_matrix_dense.sum(dim=1).to(torch.float)
    # 添加一个小的常数以防止除零错误
    degrees = degrees.clamp(min=1)
    deg_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))

    # 确保deg_inv_sqrt与adj_matrix_dense在相同的设备上
    if deg_inv_sqrt.device != adj_matrix_dense.device:
        deg_inv_sqrt = deg_inv_sqrt.to(adj_matrix_dense.device)

    # 归一化邻接矩阵
    norm_adj = deg_inv_sqrt @ adj_matrix_dense @ deg_inv_sqrt

    # 根据需要，可以选择将norm_adj转换回稀疏张量并移回GPU
    # norm_adj_sparse = torch.sparse_coo_tensor(norm_adj.coalesce().indices(), norm_adj.coalesce().values(), size=norm_adj.size()).to('cuda')

    return norm_adj


def cal_laplacian(adj_matrix):
    """
    计算图的拉普拉斯矩阵，先将稀疏张量转换为密集张量
    """
    adj_matrix_dense = adj_matrix.to_dense()  # 将稀疏张量转换为密集张量
    degree_matrix = torch.diagflat(torch.sum(adj_matrix_dense, dim=1))
    laplacian = degree_matrix - adj_matrix_dense
    return laplacian


def filter_noise_edges(adj_train_raw, num_eigenvalues):
    """
    使用拉普拉斯矩阵的特征值分解识别并移除高频率噪声边
    :param adj_train_raw: 原始邻接矩阵张量（可能为稀疏张量）
    :param num_eigenvalues: 要考虑的最小特征值数量，这些通常是低频信号，剩下的视为高频噪声
    :return: 清理后的邻接矩阵张量
    """
    laplacian = cal_laplacian(adj_train_raw)
    laplacian_cpu = laplacian.cpu().numpy()  # 先将张量从GPU移到CPU，再转换为NumPy数组
    eigenvalues, _ = eigsh(laplacian_cpu, k=num_eigenvalues + 1, which='SM')  # 计算前num_eigenvalues+1个最小特征值
    threshold = eigenvalues[-1]  # 选取最大保留特征值作为阈值

    # 将稀疏张量转换为密集张量以便进行比较和乘法操作
    adj_train_dense = adj_train_raw.to_dense().cpu()
    mask = adj_train_dense <= threshold  # 创建过滤掩码
    filtered_adj_matrix_dense = adj_train_dense * mask  # 应用过滤条件

    # 根据情况决定是否需要转换回稀疏张量
    # 注意：只有当过滤后矩阵仍然保持较稀疏时，转换为稀疏张量才有意义
    # filtered_adj_matrix_sparse = filtered_adj_matrix_dense.to_sparse()  # 示例转换回稀疏张量的语句，需评估是否适用

    return filtered_adj_matrix_dense.cuda()  # 或者直接返回密集张量并送回GPU

def save_matrices_to_csv(adj_train_list, adj_test_list, output_dir='adjacency_matrices'):
    """
    将训练和测试的邻接矩阵保存为CSV文件。

    :param adj_train_list: 训练邻接矩阵列表
    :param adj_test_list: 测试邻接矩阵列表
    :param output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (adj_train, adj_test) in enumerate(zip(adj_train_list, adj_test_list)):
        # 将稀疏矩阵转换为dense形式以便保存
        adj_train_dense = adj_train.to_dense().cpu().numpy()
        # 和这行
        adj_test_dense = adj_test.to_dense().cpu().numpy()

        # 使用Pandas保存为CSV
        pd.DataFrame(adj_train_dense).to_csv(os.path.join(output_dir, f'train_adj_matrix_view_{i + 1}.csv'),
                                             index=False)
        pd.DataFrame(adj_test_dense).to_csv(os.path.join(output_dir, f'test_adj_matrix_view_{i + 1}.csv'), index=False)


