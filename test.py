import os
import random
import argparse
import numpy as np
import torch

from dataloader import load_data
from network import MCMVC
from metric import valid
from res import MetricTracker

# ======================
# ======================
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', type=str, default='HandWritten')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--low_feature_dim', type=int, default=512)
parser.add_argument('--high_feature_dim', type=int, default=128)
parser.add_argument('--model_root_path', type=str, default='./models/')

args = parser.parse_args()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ======================
# 固定随机种子
# ======================
def setup_seed(seed=40):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

lam = 0.1
setup_seed(40)


# ======================
# 加载数据
# ======================
dataset, dims, view, data_size, class_num = load_data(args.dataset)

model_path = args.model_root_path + args.dataset +'/' + args.dataset +'_' + str(view) +'_' + str(lam) + '.pth'
# ======================
# 加载模型
# ======================
model = MCMVC(view, dims, args.low_feature_dim, args.high_feature_dim,  device)
model = model.to(device)

# ======================
# 加载训练权重
# ======================
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}")

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

print("成功加载模型权重:", model_path)

# ======================
# 开始测试
# ======================
tracker = MetricTracker()

model.eval()
with torch.no_grad():
    valid(model, device, dataset, view, data_size, class_num, tracker)

Nmi, Ari, Acc, Pur = tracker.get_result()

print("\n========== Final Test Result ==========")
print("ACC = {:.4f}".format(Acc))
print("NMI = {:.4f}".format(Nmi))
print("PUR = {:.4f}".format(Pur))
print("ARI = {:.4f}".format(Ari))
print("======================================")