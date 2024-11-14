"""
Model: GNN (a graph neural network model with 2 GCN layers)
Model Architecture
1. GCNConv layer with 16 output features
2. GCNConv layer with output features equal to number of classes
"""

import time
import torch
import torch_geometric
from dgl.data import PPIDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from ogb.nodeproppred import PygNodePropPredDataset

from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch_geometric.graphgym import optim
from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import PlanetoidG
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import torch
from torch_geometric.datasets import Planetoid, Reddit, PPI, TUDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, log_loss, classification_report
import numpy as np
import time
import psutil
import dgl
import torch.nn.functional as F

from ESGNN.test_cuda import set_gpu_memory
from load_data import get_data_info, load_dataset_by_name
from test_cuda import clean_gpu
from metis_calculation_job_GPU import estimate_task_gpu


def split_dataset(data, split_ratios=[5, 3, 2]):
    # Create custom train/validation/test splits
    num_nodes = data.num_nodes
    indices = torch.arange(num_nodes)

    # Calculate test size and validation size based on ratios
    test_size = split_ratios[2] / sum(split_ratios)  # test split
    val_size = split_ratios[1] / (split_ratios[0] + split_ratios[1])  # validation split from the remaining train+val

    # Split indices for training, validation, and testing
    # train + validation(70 %) and test(30 %)
    train_indices, test_indices = train_test_split(indices.numpy(), test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=42)

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Assign masks to the data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # print(f"Custom Training Set Size: {train_mask.sum().item()} ({train_mask.sum().item() / num_nodes:.2%})")
    # print(f"Custom Validation Set Size: {val_mask.sum().item()} ({val_mask.sum().item() / num_nodes:.2%})")
    # print(f"Custom Test Set Size: {test_mask.sum().item()} ({test_mask.sum().item() / num_nodes:.2%})")
    return data

def check_train_test(data):
    # Get masks
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask


    # Calculate proportions
    num_nodes = data.num_nodes
    num_train = train_mask.sum().item()
    num_val = val_mask.sum().item() if val_mask is not None else 0
    num_test = test_mask.sum().item()

    print(f"Total nodes: {num_nodes}")
    print(f"Number of training nodes: {num_train} ({num_train / num_nodes:.2%})")
    print(f"Number of validation nodes: {num_val} ({num_val / num_nodes:.2%})" if val_mask is not None else "Validation set not provided")
    print(f"Number of test nodes: {num_test} ({num_test / num_nodes:.2%})")


class GNN1(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN, self).__init__()
        # 使用 DGL 的 GraphConv 層
        self.conv1 = GraphConv(input_dim, 16)
        self.conv2 = GraphConv(16, output_dim)

    def forward(self, data):
        # 保持與 PyG 一樣的參數形式: data.x (節點特徵), data.edge_index (邊列表)
        x, edge_index = data.x, data.edge_index

        # 1. 將 edge_index 轉換為 DGL 圖對象
        num_nodes = x.size(0)
        src, dst = edge_index
        g = dgl.graph((src, dst), num_nodes=num_nodes)

        # 2. 添加自環以避免 0 入度節點
        g = dgl.add_self_loop(g)

        # 3. 將特徵傳遞到 DGL 的圖卷積層
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)

        # 4. 返回結果
        return F.log_softmax(x, dim=1)
        # # 使用平均池化获取图的表示
        # g.ndata['feat'] = x
        # hg = dgl.mean_nodes(g, 'feat')  # 聚合图中的节点特征
        #
        # # 分类层
        # out = self.fc(hg)
        # return F.log_softmax(out, dim=1)

# def train(model, subgraph):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     model.train()
#     optimizer.zero_grad()
#     out = model(subgraph)
#     loss = criterion(out[subgraph.train_mask], subgraph.y[subgraph.train_mask])
#     loss.backward()
#     optimizer.step()
#     return loss.item()

def train(model, subgraph):
    # print(f"min label: {subgraph.y.min()}, max label: {subgraph.y.max()}")
    # print(f"label shape: {subgraph.y[subgraph.train_mask].shape}")  # 應該是 [batch_size]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()


    model.train()
    optimizer.zero_grad()

    # 模型輸出 logits
    out = model(subgraph)

    # 獲取標籤中的最大類別值
    num_classes = subgraph.y.max().item() + 1  # 0-max => max+1 classes
    # print(f"number of labels classes: {num_classes}")

    # 當前模型輸出的類別數
    output_classes = out.shape[1]
    # print(f"number of output classes: {output_classes}")

    if output_classes < num_classes:
        # 如果輸出類別數不足，則創建額外的 logits
        batch_size = out[subgraph.train_mask].shape[0]
        extra_classes = num_classes - output_classes

        # 創建額外的非常小的 logits
        extra_logits = torch.full((batch_size, extra_classes), -1e9).to(out.device)  # 非常小的值，表示模型不確定這些類別

        # 拼接原始 logits 和擴展的 logits
        extended_output = torch.cat((out[subgraph.train_mask], extra_logits), dim=1)
        # print(f"extended model output shape: {extended_output.shape}")
    else:
        # 如果輸出類別數足夠，則不需要擴展
        extended_output = out[subgraph.train_mask]



    # 計算損失
    # loss = criterion(extended_output, subgraph.y[subgraph.train_mask])
    loss = criterion(extended_output, subgraph.y[subgraph.train_mask].squeeze())
    # loss = criterion(out, subgraph.y)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, subgraph, mask_type='test'):
    """
    accuracy: train accuracy or test accuracy
    """
    model.eval()
    _, pred = model(subgraph).max(dim=1)

    if mask_type == 'test':
        mask = subgraph.test_mask
    elif mask_type == 'train':
        mask = subgraph.train_mask
    else:
        raise ValueError("mask_type should be either 'test' or 'train'")

    # Ensure that test_mask is not empty
    mask_sum = int(mask.sum())
    if mask_sum == 0:
        print("Warning: No nodes in test_mask.")
        return 0.0

    correct = int(pred[mask].eq(subgraph.y[mask]).sum().item())
    accuracy = correct / mask_sum  # Accuracy for the specified subset

    pred = pred[mask].cpu().numpy()
    true_labels = subgraph.y[mask].cpu().numpy()
    f1 = f1_score(true_labels, pred, average='weighted', zero_division=0)

    # print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1}')

    return accuracy,f1


def train_each_epoch(model, data,epoch,epochs):
    epoch_loss = train(model, data)
    epoch_train_accuracy,f1 = test(model, data, mask_type='train')
    epoch_test_accuracy,f1 = test(model, data, mask_type='test')
    # print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_train_accuracy:.4f} - Test Accuracy: {epoch_test_accuracy:.4f}")
    return epoch_loss,epoch_train_accuracy, epoch_test_accuracy


def early_stop(patience,epoch,epoch_test_accuracy,best_test_accuracy,epochs_since_improvement):
    # Check if test accuracy improved
    if epoch_test_accuracy > best_test_accuracy:
        best_test_accuracy = epoch_test_accuracy
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    # Early stopping
    if epochs_since_improvement >= patience:
        print(f"Early stopping triggered in Epoch {epoch}.")
        return [True,epochs_since_improvement,best_test_accuracy]
        # break
    else:
        return [False,epochs_since_improvement,best_test_accuracy]

def train_model(model, data, epochs=200,patience=20,early_stopping=True,split_ratios=[6, 3, 2]):
    if len(split_ratios) == 3:
        try:
            split_dataset(data, split_ratios)
        except:
            pass

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    gpu_before=torch.cuda.memory_allocated(0) / (1024 ** 2)

    # Move model and data to the chosen device (GPU or CPU)
    train_loader = DataLoader([data], batch_size=32, shuffle=True)
    print(f"Dataset size: {len(train_loader.dataset)}")
    model = model.to(device)
    # data = data.to(device)

    train_accuracies = []
    test_accuracies = []
    losses = []
    best_test_accuracy = 0
    epochs_since_improvement = 0

    # try:
    for epoch in range(epochs):
            epoch_test_accuracy = 0
            # epoch_loss, epoch_train_accuracy, epoch_test_accuracy = train_each_epoch(model, data, epoch,epochs)
            epoch_test_accuracy = 0
            for batch in train_loader:
                # print(batch)
                batch = batch.to('cuda')
                epoch_loss, epoch_train_accuracy, epoch_test_accuracy = train_each_epoch(model, batch, epoch, epochs)
                # loss = train(model, data)
                # epoch_train_accuracy = test(model, data, mask_type='train')
                # epoch_test_accuracy = test(model, data, mask_type='test')
                # print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_train_accuracy:.4f} - Test Accuracy: {epoch_test_accuracy:.4f}")

                # 存储epoch结果：检查这些指标以确定模型是否已经收敛
                losses.append(epoch_loss)
                train_accuracies.append(epoch_train_accuracy)
                test_accuracies.append(epoch_test_accuracy)

            if early_stopping:
                [result,epochs_since_improvement, best_test_accuracy] = early_stop(patience, epoch, epoch_test_accuracy, best_test_accuracy, epochs_since_improvement)
                if result:
                    break

    # except Exception as e:
    #     print(e)
    # finally:
    if True:
        # 訓練完成後,將數據移回CPU
        gpu_after=torch.cuda.memory_allocated(0) / (1024 ** 2)
        gpu_usage = gpu_after - gpu_before
        print(f'GPU allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB')
        # print('kkkkk')
        clean_gpu()
        del gpu_after
        del gpu_before

        torch.cuda.empty_cache()

        # print('aaa')
        print(f"Edge index max value: {data.edge_index.max()}")
        assert data.edge_index.max() < data.num_nodes, "Invalid node index in edge_index."
        data = data.to("cpu")
        model = model.to("cpu")
        torch.cuda.empty_cache()


        print("Data and Model moved back to CPU")

    return losses, train_accuracies,test_accuracies,gpu_usage


def evaluate_model(model, data):
    model.eval()

    # 获取输出的概率分布
    output = model(data)
    probs = torch.nn.functional.softmax(output, dim=1)

    preds = probs.argmax(dim=1)

    # 获取测试集上的预测结果和标签
    preds = preds[data.test_mask].cpu().numpy()
    labels = data.y[data.test_mask].cpu().numpy()


    # 获取测试集上的预测概率
    probs = probs[data.test_mask].detach().cpu().numpy()

    # 确保每个样本的概率和为1
    probs = probs / probs.sum(axis=1, keepdims=True)
    # Ensure each row in probs sums to 1
    assert np.allclose(probs.sum(axis=1), np.ones(probs.shape[0])), "Probabilities do not sum to 1 for all samples!"

    # 获取所有可能的类别
    all_labels = torch.unique(data.y).cpu().numpy()
    probs = probs[:, all_labels]  # 重新对齐概率分布

    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(labels, preds)
    accuracy = accuracy_score(labels, preds)
    # precision = precision_score(labels, preds, average='weighted')
    # recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    # logloss = log_loss(labels, probs, labels=all_labels)
    class_report = classification_report(labels, preds)

    # # 将标签进行二值化，以便于计算ROC和PR曲线
    # labels_binarized = label_binarize(labels, classes=all_labels)
    #
    # # ROC and AUC for each class
    # roc_auc_dict = {}
    # pr_auc_dict = {}
    # fpr_dict = {}
    # tpr_dict = {}
    # precision_curve_dict = {}
    # recall_curve_dict = {}
    #
    # for i in range(len(all_labels)):
    #     fpr, tpr, _ = roc_curve(labels_binarized[:, i], probs[:, i])
    #     roc_auc_dict[i] = auc(fpr, tpr)
    #     fpr_dict[i] = fpr
    #     tpr_dict[i] = tpr
    #
    #     precision_curve, recall_curve, _ = precision_recall_curve(labels_binarized[:, i], probs[:, i])
    #     pr_auc_dict[i] = auc(recall_curve, precision_curve)
    #     precision_curve_dict[i] = precision_curve
    #     recall_curve_dict[i] = recall_curve

    # print("Confusion Matrix:\n", conf_matrix)
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"Log Loss: {logloss:.4f}")
    print("Classification Report:\n", class_report)

    # Plot ROC and PR curves
    # plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict)
    # plot_pr_curve(precision_curve_dict, recall_curve_dict,pr_auc_dict)

    # return {
    #     "conf_matrix": conf_matrix,
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "recall": recall,
    #     "f1_score": f1,
    #     "log_loss": logloss,
    #     "roc_auc": roc_auc_dict,
    #     "pr_auc": pr_auc_dict,
    #     "fpr": fpr_dict,
    #     "tpr": tpr_dict,
    #     "precision_curve": precision_curve_dict,
    #     "recall_curve": recall_curve_dict,
    #     "class_report": class_report
    # }


# Plot ROC Curve
def plot_roc_curve(fpr_dict, tpr_dict, roc_auc_dict):
    plt.figure()
    lw = 2

    # 对每一个类别绘制ROC曲线
    for i in fpr_dict:
        plt.plot(fpr_dict[i], tpr_dict[i], lw=lw, label=f'Class {i} ROC curve (area = {roc_auc_dict[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()



# Plot PR Curve
def plot_pr_curve(precision_dict, recall_dict, pr_auc_dict):
    plt.figure()
    lw = 2

    # 对每一个类别绘制PR曲线
    for i in precision_dict:
        plt.plot(recall_dict[i], precision_dict[i], lw=lw,
                 label=f'Class {i} PR curve (area = {pr_auc_dict[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower right")
    plt.show()



def plot_epochs_acc(train_accuracies,test_accuracies):
    # Plot epochs training and testing accuracy
    plt.plot(range(1, len(test_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy Over Epochs')
    plt.legend()
    plt.show()


def plot_epochs_loss(losses):
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'g', label='Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def measure_time_and_memory(model, data,epochs=200, patience=20, early_stopping=True,split_ratios=[6,3,2],is_save=False):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model and data to the chosen device (GPU or CPU)
    model = model.to(device)
    data = data.to(device)
    measure_info = []

    if device.type == 'cuda':
        # GPU execution
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Start recording time
        start_event.record()

        # Run the model (assume one forward pass for the example)
        train_model(model, data, epochs=epochs, patience=patience, early_stopping=early_stopping,split_ratios=split_ratios)
        # output = model(data)

        # End recording time
        end_event.record()

        # Wait for all operations to finish
        torch.cuda.synchronize()

        # Compute total runtime in seconds
        total_time = start_event.elapsed_time(end_event) / 1000  # convert to seconds

        # Get GPU memory usage
        gpu_memory_used = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
        max_gpu_memory_used = torch.cuda.max_memory_allocated() / 1024 ** 2  # in MB

        print(f"Runtime on GPU: {total_time:.4f} seconds")
        print(f"GPU Memory Used: {gpu_memory_used:.2f} MB")
        print(f"Max GPU Memory Used: {max_gpu_memory_used:.2f} MB")

        measure_info.append(f"Time: {total_time:.4f} seconds")
        measure_info.append(f"GPU Memory: {gpu_memory_used:.2f} MB")
        measure_info.append(f"Max GPU Memory: {max_gpu_memory_used:.2f} MB")

        if is_save:
            # 将输出内容写入文件
            with open('gpu_measure_result.txt', 'a') as f:
                for line in measure_info:
                    line = "".join(map(str, line))
                    f.write(line + '\n')
                f.write('--------------------------------------\n')

    else:
        # CPU execution
        start_time = time.time()

        # Run the model (assume one forward pass for the example)
        train_model(model, data, epochs=epochs, patience=patience, early_stopping=early_stopping,
                    split_ratios=split_ratios)
        output = model(data)

        # End time
        end_time = time.time()

        # Calculate runtime
        total_time = end_time - start_time

        # Get CPU memory usage
        process = psutil.Process()
        cpu_memory_used = process.memory_info().rss / 1024 ** 2  # in MB

        print(f"Runtime on CPU: {total_time:.4f} seconds")
        print(f"CPU Memory Used: {cpu_memory_used:.2f} MB")

        measure_info.append(f"Time: {total_time:.4f} seconds")
        measure_info.append(f"CPU Memory: {cpu_memory_used:.2f} MB")


        if is_save:
            # 将输出内容写入文件
            with open('cpu_measure_result.txt', 'a') as f:
                for line in measure_info:
                    line = "".join(map(str, line))
                    f.write(line + '\n')
                f.write('--------------------------------------\n')

def measure_time_and_memory_dataset(dataset):
    get_data_info(dataset)
    data = dataset[0]
    print(data)
    # input_dim = dataset[0].x.size(1)  # Feature dimension from the first subgraph
    # output_dim = len(torch.unique(dataset[0].y))  # Number of classes based on the labels in the first subgraph
    # model = GNN(input_dim, output_dim)
    # train_model(model, data, epochs=200, patience=20, early_stopping=True, split_ratios=[6, 3, 2])
    # measure_time_and_memory(model, data, epochs=200, patience=20, early_stopping=True, split_ratios=[6, 3, 2],is_save=False)

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # 创建 GNN 模型
    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    model = GNN(input_dim=input_dim, output_dim=output_dim)

    # 开始训练
    train1(model, train_loader, epochs=20)




# 将 PyG 的 Data 对象转换为 DGL 图
def pyg_to_dgl(data):
    x, edge_index, y = data.x, data.edge_index, data.y
    num_nodes = x.size(0)
    src, dst = edge_index
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)
    g.ndata['feat'] = x
    g.ndata['label'] = y
    return g

# 训练函数
def train1(model, dataloader, epochs=20, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            g = pyg_to_dgl(batch)
            out = model(batch)
            labels = batch.y.float()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model

if __name__ == '__main__':
    # dataset = TUDataset(root='./dataset/TUDataset/PROTEINS', name='PROTEINS')
    # measure_time_and_memory_dataset(dataset)
    #
    # measure_time_and_memory(model, data, epochs=200, patience=20, early_stopping=True, split_ratios=[6, 3, 2],
    #                         is_save=False)
    # datasets = ['PPI', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY']
    datasets = ['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'Amazon-Computers','Amazon-Photo']
    # datasets = ['Reddit']
    # datasets = ['PPI', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY']
    # datasets = ['ogbn-products', 'ogbn-proteins', 'ogbn-arxiv']

    for dataset_name in datasets:
        dataset = load_dataset_by_name(dataset_name)
        get_data_info(dataset)
        data = dataset[0]
        print(data)
        # 将图数据转换为NetworkX图
        # G = to_networkx(data, to_undirected=True)

        # split_dataset(data, split_ratios=[6, 3, 2])

        measure_time_and_memory_dataset(dataset)

            # input_dim = dataset[0].x.size(1)  # Feature dimension from the first subgraph
            # output_dim = len(torch.unique(dataset[0].y))  # Number of classes based on the labels in the first subgraph
            # model = GNN(input_dim, output_dim)
            # # set_gpu_memory(8000)
            #
            # # data.to('cuda')
            # # model.to('cuda')
            # # print(torch.cuda.memory_summary(device='cuda'))
            #
            # # Train and test the model
            # # losses,train_accuracies,test_accuracies = train_model(model, data ,epochs=200,patience=20,early_stopping=True,split_ratios=[6,3,2])
            # #
            # # evaluate_model(model,data)
            # # gpu_memory_MB, model_time=estimate_task_gpu(model, data)
            # # print(gpu_memory_MB, model_time)
            # measure_time_and_memory(model, data, epochs=200, patience=20, early_stopping=True, split_ratios=[6, 3, 2],
            #                         is_save=False)
        # except Exception as e:
        #     print(f"Error running es_gpu on {dataset}: {e}")






