import os

import torch


def get_all_file_paths(directory):
    pt_file_paths = []
    pt_folder_paths = set()  # 使用 set 防止重复
    # os.walk 会递归遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):  # 检查文件是否以 .pt 结尾
                # 获取每个文件的完整路径
                pt_file_paths.append(os.path.join(root, file))
                pt_folder_paths.add(root)  # 只保存文件夹路径
    return pt_folder_paths


def load_subgraphs(dir_path, num_subgraphs):
    subgraphs = []
    for i in range(num_subgraphs):
        subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
        subgraphs.append(subgraph)
    return subgraphs

def get_subgraphs_from_dir(directory = 'subgraph_data', num_subgraphs=4):
    all_pt_folder_paths = get_all_file_paths(directory)
    for pt_folder_path in all_pt_folder_paths:
        print(pt_folder_path)
        subgraphs = load_subgraphs(pt_folder_path,num_subgraphs)
        print(subgraphs)

if __name__ == '__main__':
    directory = 'subgraph_data'
    num_subgraphs = 4
    all_pt_folder_paths = get_all_file_paths(directory)
    print(len(all_pt_folder_paths))
    for pt_folder_path in all_pt_folder_paths:
        print(pt_folder_path)
        subgraphs = load_subgraphs(pt_folder_path, num_subgraphs)
        print(subgraphs)
    # get_subgraphs_from_dir(num_subgraphs=4)

