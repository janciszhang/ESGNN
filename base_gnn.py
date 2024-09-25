"""
Model: GNN (a graph neural network model with 2 GCN layers)
Model Architecture
1. GCNConv layer with 16 output features
2. GCNConv layer with output features equal to number of classes
"""

import time
import torch
import torch_geometric
from sklearn.preprocessing import label_binarize
# from torch_geometric.datasets import PlanetoidG
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import torch
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, log_loss, classification_report
import numpy as np


def split_dataset(data):
    # Create custom train/validation/test splits
    num_nodes = data.num_nodes
    indices = torch.arange(num_nodes)

    # Split indices for training, validation, and testing
    # train + validation(70 %) and test(30 %)
    train_indices, test_indices = train_test_split(indices.numpy(), test_size=0.3, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

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


class GNN(torch.nn.Module):
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

def train(model, subgraph):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    out = model(subgraph)
    loss = criterion(out[subgraph.train_mask], subgraph.y[subgraph.train_mask])
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

    return accuracy,f1


def train_each_epoch(model, data,epoch,epochs):
    epoch_loss = train(model, data)
    epoch_train_accuracy,f1 = test(model, data, mask_type='train')
    epoch_test_accuracy,f1 = test(model, data, mask_type='test')
    # print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_train_accuracy:.4f} - Test Accuracy: {epoch_test_accuracy:.4f}")
    return epoch_loss,epoch_train_accuracy, epoch_test_accuracy


def train_epochs(model, data, epochs=200,patience=20,early_stopping=True):
    train_accuracies = []
    test_accuracies = []
    losses = []
    best_test_accuracy = 0
    epochs_since_improvement = 0

    for epoch in range(epochs):
        epoch_loss, epoch_train_accuracy, epoch_test_accuracy = train_each_epoch(model, data, epoch,epochs)
        # loss = train(model, data)
        # epoch_train_accuracy = test(model, data, mask_type='train')
        # epoch_test_accuracy = test(model, data, mask_type='test')
        # print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_train_accuracy:.4f} - Test Accuracy: {epoch_test_accuracy:.4f}")

        # 存储epoch结果：检查这些指标以确定模型是否已经收敛
        losses.append(epoch_loss)
        train_accuracies.append(epoch_train_accuracy)
        test_accuracies.append(epoch_test_accuracy)

        if early_stopping:
            # Check if test accuracy improved
            if epoch_test_accuracy > best_test_accuracy:
                best_test_accuracy = epoch_test_accuracy
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Early stopping
            if epochs_since_improvement >= patience:
                # print(f"Early stopping triggered in Epoch {epoch}.")
                break

    return losses, train_accuracies,test_accuracies


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


if __name__ == '__main__':
    # 加载Cora数据集
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # 获取图数据和标签
    data = dataset[0]
    # 将图数据转换为NetworkX图
    # G = to_networkx(data, to_undirected=True)

    split_dataset(data)
    model = GNN(dataset)

    # Train and test the model
    epochs = 200

    losses,train_accuracies,test_accuracies = train_epochs(model, data ,epochs)

    evaluate_model(model,data)

