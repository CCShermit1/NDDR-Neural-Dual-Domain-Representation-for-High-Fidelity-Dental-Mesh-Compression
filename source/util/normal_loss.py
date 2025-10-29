import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_normal_loss(pred_normals, true_normals, lambda_weight=0.1):

    pred_normals = pred_normals.detach().cpu()
    true_normals = true_normals.detach().cpu()
    # 计算距离矩阵
    distance_matrix = np.linalg.norm(pred_normals[:, np.newaxis] - true_normals[np.newaxis, :], axis=2)
    
    # 使用匈牙利算法找到最优匹配
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    
    # 计算匹配的损失
    matched_loss = np.sum(np.square(distance_matrix[row_indices, col_indices]))

    # 计算未匹配的预测法线损失
    matched_rows = set(row_indices)
    extra_loss = np.sum(np.square(np.linalg.norm(pred_normals, axis=1)**2)) - np.sum(np.square(np.linalg.norm(pred_normals[list(matched_rows)], axis=1)**2))

    # 综合损失
    total_loss = matched_loss + lambda_weight * extra_loss

    return total_loss


def compute_loss_in_batches(pred_normals, true_normals, batch_size, lambda_weight=0.1):
    total_loss = 0
    num_batches = int(np.ceil(len(pred_normals) / batch_size))
    
    for i in range(num_batches):
        batch = pred_normals[i * batch_size: (i + 1) * batch_size]
        loss = compute_normal_loss(batch, true_normals, lambda_weight)
        total_loss += loss
    
    return total_loss

# # 示例用法
# pred_normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 预测法线
# true_normals = np.array([[1, 0, 0], [0, 1, 0]])  # 真值法线

# loss = compute_loss(pred_normals, true_normals)
# print("总损失:", loss)
