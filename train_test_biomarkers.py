""" Training and testing of the model
"""
import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from utils import one_hot_tensor, cal_sample_weight
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv


cuda = True if torch.cuda.is_available() else False  # 判断是否有GPU可用
torch.autograd.set_detect_anomaly(True)
""" Componets of the model
"""
# 定义xavier初始化函数
def xavier_init(m):
    if type(m) == nn.Linear:  # 如果是线性层
        nn.init.xavier_normal_(m.weight)  # 对线性层的权重使用xavier正态分布初始化
        if m.bias is not None:  # 如果线性层有偏置
           m.bias.data.fill_(0.0)  # 对线性层偏置初始化为0

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight.clone())  # 使用 clone() 以避免就地操作
        if torch.cuda.is_available():
            adj = adj.to('cuda')
        support = support.to(adj.device)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias.clone()  # 使用 clone() 以避免就地操作
        else:
            return output


class ViewAttention(nn.Module):
    def __init__(self, in_dim, num_view):
        super(ViewAttention, self).__init__()
        self.num_view = num_view
        self.linear_layers = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(num_view)])
        self.attention_weights = nn.Parameter(torch.FloatTensor(num_view, in_dim))
        nn.init.xavier_normal_(self.attention_weights.data)

    def forward(self, view_features):
        transformed_features = [self.linear_layers[i](view_features[i]) for i in range(self.num_view)]
        attention_scores = []
        for i in range(self.num_view):
            score = torch.sum(transformed_features[i] * self.attention_weights[i], dim=1, keepdim=True)
            attention_scores.append(score)
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_weights = F.softmax(attention_scores.clone(), dim=1)  # 使用 clone() 以避免就地操作

        # 加权特征组合
        weighted_features = 0
        for i in range(self.num_view):
            weighted_features += transformed_features[i] * attention_weights[:, i].unsqueeze(-1)

        # 每个视图最终特征 = 原视图特征 + 其他视图加权特征
        final_features = []
        for i in range(self.num_view):
            final_feature = view_features[i] + weighted_features - transformed_features[i] * attention_weights[:,
                                                                                             i].unsqueeze(-1)
            final_features.append(final_feature)

        return final_features


class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x.clone(), self.dropout, training=self.training)  # 使用 clone() 以避免就地操作
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x.clone(), self.dropout, training=self.training)  # 使用 clone() 以避免就地操作
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        return x

# 定义分类器
class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 该分类器包含一个全连接层
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        # 对分类器进行xavier初始化
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        input_dim = num_view * hvcdn_dim  # 修改输入维度为视图数量乘以每个视图的特征维度
        self.model = nn.Sequential(
            nn.Linear(input_dim, hvcdn_dim),  # 确保输入维度正确
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)  # 调用独立的xavier_init函数

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = in_list[i].view(in_list[i].size(0), -1)  # 展平特征
            in_list[i] = torch.sigmoid(in_list[i])  # 对展平特征应用sigmoid
        x = torch.cat(in_list, dim=1)  # 将所有视图特征沿第二个维度连接
        output = self.model(x)
        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, gcn_dropout=0.5):
    model_dict = {}

    if num_view == 1:
        for i in range(num_view):
            model_dict["E{:}".format(i + 1)] = GCN_E(dim_list[i], dim_he_list, gcn_dropout)
            model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        for i in range(num_view):
            model_dict["E{:}".format(i + 1)] = GCN_E(dim_list[i], dim_he_list, gcn_dropout)
            model_dict["attention"] = ViewAttention(dim_he_list[-1], num_view)
            model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
            model_dict["C"] = VCDN(num_view, num_class, dim_he_list[-1])

    # 将模型迁移到 GPU 上（如果可用）
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    return model_dict


# 定义优化器初始化函数
def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):  # 对每个视图，将编码器和分类器的参数放入Adam优化器中进行优化
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i + 1)].parameters()) + list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e)
    if num_view >= 2:  # 如果视图数大于等于2，则将VCDN模型的参数也放入Adam优化器中进行优化
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict  # 返回优化器字典



def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=',').astype(int)

    data_list = []
    for view in view_list:
        data_list.append(np.loadtxt(os.path.join(data_folder, f"{view}_features.csv"), delimiter=','))

    data_tensor_list = [torch.FloatTensor(data) for data in data_list]
    if cuda:
        data_tensor_list = [data_tensor.cuda() for data_tensor in data_tensor_list]

    return data_tensor_list, labels

def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_train_list = []
    adj_test_list = []

    for i in range(len(data_tr_list)):
        adj_train = gen_density_adj_mat_tensor(data_tr_list[i], adj_parameter)
        adj_train_normalized = cal_normalized_laplacian(adj_train)
        adj_test = gen_density_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter)
        adj_test_normalized = cal_normalized_laplacian(adj_test)
        adj_train_list.append(torch.tensor(adj_train_normalized, dtype=torch.float).to('cuda'))
        adj_test_list.append(torch.tensor(adj_test_normalized, dtype=torch.float).to('cuda'))

    return adj_train_list, adj_test_list

def gen_density_adj_mat_tensor(data, adj_parameter):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # 计算协方差矩阵及其逆矩阵，添加小的正则化项
    cov_matrix = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-10
    inv_cov_matrix = inv(cov_matrix)

    # 计算马氏距离
    distances = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            distances[i, j] = mahalanobis(data[i], data[j], inv_cov_matrix)

    k = int(adj_parameter)
    densities = np.zeros(distances.shape[0])
    for i in range(distances.shape[0]):
        nearest_neighbors = np.argsort(distances[i])[1:k+1]  # 排除自身
        densities[i] = np.mean(distances[i][nearest_neighbors])

    adj_matrix = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[i])
        threshold = densities[i] * adj_parameter
        edges_added = 0
        for j in sorted_indices:
            if i != j:
                if distances[i, j] <= threshold or edges_added < k:
                    adj_matrix[i, j] = 1 / (1 + distances[i, j])
                    edges_added += 1
                else:
                    break

    return adj_matrix

def gen_density_test_adj_mat_tensor(data, trte_idx, adj_parameter):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # 计算协方差矩阵及其逆矩阵，添加小的正则化项
    cov_matrix = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-10
    inv_cov_matrix = inv(cov_matrix)

    # 计算马氏距离
    distances = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            distances[i, j] = mahalanobis(data[i], data[j], inv_cov_matrix)

    k = int(adj_parameter)
    densities = np.zeros(distances.shape[0])
    for i in range(distances.shape[0]):
        nearest_neighbors = np.argsort(distances[i])[1:k+1]
        densities[i] = np.mean(distances[i][nearest_neighbors])

    adj_matrix = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[i])
        threshold = densities[i] * adj_parameter
        edges_added = 0
        for j in sorted_indices:
            if i != j:
                if distances[i, j] <= threshold or edges_added < k:
                    adj_matrix[i, j] = 1 / (1 + distances[i, j])
                    edges_added += 1
                else:
                    break

    return adj_matrix

def cal_normalized_laplacian(adj):
    # 计算归一化拉普拉斯矩阵
    D = np.sum(adj, axis=1)
    D[D == 0] = 1e-10  # 对于度数为零的节点，赋予一个非常小的值
    D_inv_sqrt = 1.0 / np.sqrt(D)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0  # 将无穷大值设为0
    D_inv_sqrt[np.isnan(D_inv_sqrt)] = 0.0  # 将NaN值设为0
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)  # 使用中间数组创建对角矩阵
    L = np.eye(adj.shape[0]) - np.dot(np.dot(D_inv_sqrt_mat, adj), D_inv_sqrt_mat)
    return L

def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)

    # 提取每个视图的特征
    view_features = []
    for i in range(num_view):
        features = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
        view_features.append(features.clone())

    # 通过注意力机制生成加权特征
    if num_view >= 2:
        combined_features = model_dict["attention"](view_features)
    else:
        combined_features = view_features

    # 使用Classifier_1对每个视图特征进行独立分类
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)].zero_grad()
        ci = model_dict["C{:}".format(i + 1)](combined_features[i].clone())
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward(retain_graph=True)
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()

    # 通过VCDN进行联合分类（如果视图数大于等于2）
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        flattened_features = [f.view(f.size(0), -1) for f in combined_features]
        c = model_dict["C"](flattened_features)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)

    view_features = []
    for i in range(num_view):
        features = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
        view_features.append(features.clone())

    if num_view >= 2:
        combined_features = model_dict["attention"](view_features)
    else:
        combined_features = view_features[0]

    if num_view >= 2:
        flattened_features = [f.view(f.size(0), -1) for f in view_features]
        c = model_dict["C"](flattened_features)
    else:
        c = combined_features

    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    return prob


def train_test_with_feature_importance(data_folder, view_list, num_class,
                                       lr_e_pretrain, lr_e, lr_c,
                                       num_epoch_pretrain, num_epoch, adj_parameter, dim_he_list,
                                       compute_feat_imp=False, topn=30):
    # 读取特征名称
    featname_list = []
    for v in view_list:
        featname_file = os.path.join(data_folder, f"{v}_featname.csv")
        df = pd.read_csv(featname_file, header=None)
        featname_list.append(df.values.flatten())

    data_tensor_list, labels_all = prepare_trte_data(data_folder, view_list)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    best_accs = []
    best_f1_weighted_scores = []
    best_bacc_scores = []
    mean_epoch_results = {'accs': [], 'f1s': [], 'baccs': []}
    feature_importance_all = []
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    for train_idx, test_idx in kf.split(np.zeros(len(labels_all)), labels_all):
        # 训练和测试集
        train_data_list = [data_tensor[train_idx].clone() for data_tensor in data_tensor_list]
        test_data_list = [data_tensor[test_idx].clone() for data_tensor in data_tensor_list]
        labels_train = labels_all[train_idx]
        labels_test = labels_all[test_idx]

        labels_tr_tensor = torch.LongTensor(labels_train)
        onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
        sample_weight_tr = cal_sample_weight(labels_train, num_class)
        sample_weight_tr = torch.FloatTensor(sample_weight_tr)
        print('data prepared.\n')
        if cuda:
            labels_tr_tensor = labels_tr_tensor.cuda()
            onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
            sample_weight_tr = sample_weight_tr.cuda()
        print('building...\n')
        adj_tr_list, adj_te_list = gen_trte_adj_mat(train_data_list, test_data_list,
                                                    {'tr': list(range(len(train_idx))),
                                                     'te': list(range(len(train_idx), len(train_idx) + len(test_idx)))},
                                                    adj_parameter)
        dim_list = [x.shape[1] for x in train_data_list]
        model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, gcn_dropout=0.5)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()

        optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
        print('pretraining...\n')
        for epoch in range(num_epoch_pretrain):
            train_epoch(train_data_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)

        optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
        best_fold_results = {'acc': 0, 'f1_weighted': 0, 'bacc': 0}
        print('training...\n')
        for epoch in range(num_epoch):
            train_epoch(train_data_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

            # 测试模型
            te_prob = test_epoch(test_data_list, adj_te_list, list(range(len(test_idx))), model_dict)
            te_pred = te_prob.argmax(1)
            te_true = labels_test

            acc = accuracy_score(te_true, te_pred)
            f1_weighted = f1_score(te_true, te_pred, average='weighted')
            bacc = balanced_accuracy_score(te_true, te_pred)

            # Update best results if current epoch's results are better
            if acc > best_fold_results['acc']:
                best_fold_results['acc'] = acc
                best_fold_results['f1_weighted'] = f1_weighted
                best_fold_results['bacc'] = bacc

            mean_epoch_results['accs'].append(acc)
            mean_epoch_results['f1s'].append(f1_weighted)
            mean_epoch_results['baccs'].append(bacc)

        best_accs.append(best_fold_results['acc'])
        best_f1_weighted_scores.append(best_fold_results['f1_weighted'])
        best_bacc_scores.append(best_fold_results['bacc'])

        #
        # print('calculating features importance...')
        # # Calculate feature importance after model training
        # if compute_feat_imp:
        #     original_f1 = f1_weighted  # Use the F1 score of the whole view as the baseline
        #     for v in range(num_view):
        #         importance_scores = np.zeros(dim_list[v])
        #         for j in range(dim_list[v]):
        #             # Save the original feature
        #             original_feature_test = test_data_list[v][:, j].clone()
        #
        #             # Zero out the feature
        #             test_data_list[v][:, j] = 0
        #
        #             # Test the model with the perturbed feature
        #             te_prob = test_epoch(test_data_list, adj_te_list, list(range(len(test_idx))), model_dict)
        #             te_pred = te_prob.argmax(1)
        #             perturbed_f1 = f1_score(te_true, te_pred, average='weighted')
        #
        #             # Calculate feature importance
        #             importance_scores[j] = original_f1 - perturbed_f1
        #
        #             # Restore the original feature
        #             test_data_list[v][:, j] = original_feature_test
        #
        #         # Store feature importance and names to feature_importance_all
        #         feature_importance_all.extend(zip(featname_list[v], importance_scores))

    print("Best Accuracy per fold: ", best_accs)
    print("Best F1 weighted per fold: ", best_f1_weighted_scores)
    print("Best BACC per fold: ", best_bacc_scores)
    print("Average Accuracy: {:.3f}".format(np.mean(best_accs)))
    print("Average F1 weighted: {:.3f}".format(np.mean(best_f1_weighted_scores)))
    print("Average BACC: {:.3f}".format(np.mean(best_bacc_scores)))
    #
    # # 输出所有视图中最重要的特征
    # if compute_feat_imp:
    #     # 将所有特征的重要性合并到一个DataFrame中
    #     feature_importance_df = pd.DataFrame(feature_importance_all, columns=["Feature", "Importance"])
    #     # 对特征重要性进行排序
    #     feature_importance_df = feature_importance_df.groupby("Feature")["Importance"].mean().reset_index()
    #     feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    #     top_features = feature_importance_df.head(topn)
    #
    #     print("Top features across all views:")
    #     for rank, row in top_features.iterrows():
    #         print(f"  Rank {rank + 1}: {row['Feature']} with importance {row['Importance']:.7f}")
