from train_test_biomarkers import train_test_with_feature_importance

if __name__ == "__main__":

    data_folders = ['ROSMAP']
    adj_parameters = [2]
    view_lists = [[1, 2, 3]]

    dim_he_list = [128, 256, 256]

    num_epoch_pretrain = 500
    num_epoch = 1000
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    for data_folder in data_folders:
        if data_folder.startswith('BRCA'):
            num_class = 5
        if data_folder.startswith('ROSMAP'):
            num_class = 2
        if data_folder.startswith('KIPAN'):
            num_class = 3
        if data_folder.startswith('LGG'):
            num_class = 2

    for adj_parameter in adj_parameters:
        for data_folder in data_folders:
            for view_list in view_lists:
                train_test_with_feature_importance(data_folder, view_list, num_class,
                                                   lr_e_pretrain, lr_e, lr_c,
                                                   num_epoch_pretrain, num_epoch, adj_parameter, dim_he_list,
                                                   compute_feat_imp=True, topn=30)

