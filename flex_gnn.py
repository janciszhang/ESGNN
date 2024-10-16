import torch
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch.optim as optim
from ESGNN.base_gnn import evaluate_model, train_model, test, split_dataset, measure_time_and_memory


# 定义 GCN 模型
class FlexibleGNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim=64):
        super(FlexibleGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

def combine_FlexibleGNN_models(model1, model2):
    # Create a new model with the same architecture
    combine_model = FlexibleGNN(input_dim=model1.convs[0].in_channels,
                                output_dim=model1.convs[-1].out_channels,
                                num_layers=len(model1.convs))

    # Average the weights of the two models
    with torch.no_grad():
        for param1, param2, param_combined in zip(model1.parameters(), model2.parameters(), combine_model.parameters()):
            param_combined.copy_((param1.data + param2.data) / 2)

    return combine_model


# 主训练和测试过程函数
def run_gnn(dataset, hidden_dim=64, num_layers=3, epochs=200):
    data = dataset[0]

    split_dataset(data, split_ratios=[6, 3, 2])

    # 处理无特征图的情况，使用度数作为节点特征
    if data.x is None:
        data.x = data.adj_t.sum(dim=1).unsqueeze(1)  # 添加维度以适应输入特征格式

    # 模型参数
    input_dim = data.x.shape[1]
    output_dim = dataset.num_classes

    # 初始化模型和优化器
    model = FlexibleGNN(input_dim, output_dim, num_layers, hidden_dim)

    # 训练模型
    print("Training model...")
    losses, train_accuracies, test_accuracies=train_model(model, data ,epochs=epochs,patience=20,early_stopping=True,split_ratios=[6,3,2])

    # 测试模型
    print("Testing model...")
    accuracy,f1=test(model, data, mask_type='test')
    print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1}')

    # 评估模型
    print("Evaluating model...")
    evaluate_model(model, data)




if __name__ == '__main__':
    # 加载 Cora 数据集
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    split_dataset(data, split_ratios=[6, 3, 2])

    # 处理无特征图的情况，使用度数作为节点特征
    if data.x is None:
        data.x = data.adj_t.sum(dim=1).unsqueeze(1)  # 添加维度以适应输入特征格式

    # 模型参数
    input_dim = data.x.shape[1]
    output_dim = dataset.num_classes
    num_layers = 3
    epochs = 200
    # 初始化模型和优化器
    model = FlexibleGNN(input_dim, output_dim, num_layers)

    # run_gnn(dataset, hidden_dim, num_layers, epochs=epochs)

    measure_time_and_memory(model, data)



