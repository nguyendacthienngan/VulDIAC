import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
def temperature_scaled_kl(p, q, T=2.0):
    p = F.log_softmax(p / T, dim=1)
    q = F.softmax(q / T, dim=1)
    return F.kl_div(p, q, reduction='batchmean') * T * T


def dynamic_weighting(loss_causal, loss_spurious, loss_context):
    # total_loss = loss_causal + loss_spurious + loss_context 
    weights = torch.tensor([loss_causal.item(), loss_spurious.item(), loss_context.item()])
    weights = F.softmax(1 / weights, dim=0)  # 权重反比于损失大小
    return weights[0] * loss_causal + weights[1] * loss_spurious + weights[2] * loss_context


def contrastive_loss(causal_features, spurious_features, margin=2.0):
    distance = torch.norm(causal_features - spurious_features, dim=1)
    # print(distance)
    loss = F.relu(margin - distance)
    return torch.mean(loss)


def contrastive_loss_cosine(causal_features, spurious_features, margin=0.4):
    distance = torch.norm(causal_features - spurious_features, dim=1)
    # print(distance)
    loss = F.relu(margin - distance)
    return torch.mean(loss)


def calculate_line_scores(casual_att, line_numbers, aggregation="sum"):
    """
    根据节点的因果注意力分数计算每一行代码的分数。
    Args:
        casual_att (list or np.ndarray): 节点的因果注意力分数，形状为 [num_nodes]。
        line_numbers (list): 节点对应的代码行号，长度为 [num_nodes]。
        aggregation (str): 对每一行代码的节点分数的聚合方式，可选 "mean"、"max"、"sum"。
    
    Returns:
        dict: 每一行代码的分数，键为行号，值为分数。
    """
    from collections import defaultdict

    # 初始化一个默认字典，用于存储每行代码的节点分数列表
    line_scores = defaultdict(list)

    # 遍历节点的分数和对应的行号
    for score, line_number in zip(casual_att, line_numbers):
        line_scores[int(line_number)].append(score)

    # 对每行代码的分数进行聚合
    aggregated_scores = {}
    for line, scores in line_scores.items():
        if aggregation == "mean":
            aggregated_scores[line] = sum(scores) / len(scores)  # 平均值
        elif aggregation == "max":
            aggregated_scores[line] = max(scores)  # 最大值
        elif aggregation == "sum":
            aggregated_scores[line] = sum(scores)  # 总和
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

    # 将聚合结果按行号排序并返回
    return sorted(aggregated_scores.items(), key=lambda x: x[0])  # 按行号排序


def get_line_ranking(all_lines_score):
    """
    根据预测分数对行进行排名，从高到低。

    Args:
        all_lines_score (list): 每行的预测分数。

    Returns:
        list: 按分数从高到低排列的行索引。
    """
    return [line[0] for line in sorted(all_lines_score, key=lambda x: x[1], reverse=True)]


def calculate_top_k_recall(ranking, flaw_line_indices, all_lines_score, top_k_loc):
    """
    计算 Top-K Recall 和 IFA 指标。

    Args:
        ranking (list): 行排名列表。
        flaw_line_indices (list): 漏洞行索引。
        all_lines_score (list): 每行的分数。
        top_k_loc (list): 用于 Recall 的比例列表。

    Returns:
        tuple: 包括 Top-K Recall 结果、最少和最多的干净行数。
    """
    correctly_predicted_flaw_lines = []
    all_clean_lines_inspected = []
    ifa_calculated = False

    for top_k in top_k_loc:
        k = int(len(all_lines_score) * top_k)  # 计算当前 Top-K 的范围
        correct_count = 0

        for flaw_idx in flaw_line_indices:
            if flaw_idx in ranking[:k]:
                correct_count += 1
                if not ifa_calculated:
                    flaw_rank_idx = ranking.index(flaw_idx)
                    all_clean_lines_inspected.append(flaw_rank_idx)

        correctly_predicted_flaw_lines.append(correct_count)
        ifa_calculated = True  # 只计算一次 IFA

    # # 计算 IFA 和 Effort 的指标
    min_clean_lines = min(all_clean_lines_inspected, default=0)
    max_clean_lines = max(all_clean_lines_inspected, default=0)

    return correctly_predicted_flaw_lines, min_clean_lines, max_clean_lines


def calculate_top_k_accuracy(ranking, flaw_line_indices, top_k_constant, filename):
    """
    计算 Top-K Accuracy 指标。

    Args:
        ranking (list): 行排名列表。
        flaw_line_indices (list): 漏洞行索引。
        top_k_constant (list): 用于 Accuracy 的常数列表。
        filename (str): 当前样本的文件名。

    Returns:
        tuple: 包括 Top-K Accuracy 结果、正确和错误定位的样本索引。
    """
    correctly_localized = []  # 用于记录每个 k 的定位结果
    correct_idx = []          # 正确定位的样本文件名
    incorrect_idx = []        # 错误定位的样本文件名

    for k in top_k_constant:
        correctly_detected = False  # 每次 k 开始时重置标志

        # 检查 flaw_line_indices 是否在 top-k 排名中
        for flaw_idx in flaw_line_indices:
            if flaw_idx in ranking[:k]:
                correctly_localized.append(1)
                correctly_detected = True
                break  # 已经找到正确结果，跳出内层循环

        if correctly_detected:
            if filename not in correct_idx:
                correct_idx.append(filename)
        else:
            if filename not in incorrect_idx:
                incorrect_idx.append(filename)
            correctly_localized.append(0)  # 如果未检测到，标记为 0

    return correctly_localized, correct_idx, incorrect_idx


def localize_evaluation(
        all_lines_score: list,
        flaw_line_indices: list,
        top_k_loc: list,
        top_k_constant: list,
        true_positive_only: bool,
        filename: str
    ):
    """
    行级漏洞检测评估函数
    
    Args:
        all_lines_score (list): 每行的预测分数
        flaw_line_indices (list): 实际漏洞行的索引
        top_k_loc (list): 用于 Top-K Recall 的比例列表
        top_k_constant (list): 用于 Top-K Accuracy 的常数列表
        true_positive_only (bool): 是否仅针对预测正确的样本进行评估
        index (int, optional): 当前样本索引（可用于标记）。默认值为 None。
    Returns:
        如果 true_positive_only=True，返回多个评估指标；否则返回所有行的分数和标签。
    """
    if true_positive_only:
        # Step 1: 计算行排名
        ranking = get_line_ranking(all_lines_score)
        # print(ranking)
        # Step 2: 统计基础信息
        total_lines = len(all_lines_score)
        num_of_flaw_lines = len(flaw_line_indices)

        # Step 3: 计算 Top-K Recall
        recall_results, min_clean_lines, max_clean_lines = calculate_top_k_recall(
            ranking, flaw_line_indices, all_lines_score, top_k_loc
        )

        # Step 4: 计算 Top-K Accuracy
        accuracy_results, correct_idx, incorrect_idx = calculate_top_k_accuracy(
            ranking, flaw_line_indices, top_k_constant, filename
        )

        # 返回评估结果
        return {
            "total_lines": total_lines,
            "num_of_flaw_lines": num_of_flaw_lines,
            "all_line_scores": all_lines_score,
            "ranking": ranking,
            "flaw_line_indices": flaw_line_indices,
            "recall_results": recall_results,
            "min_clean_lines_inspected": min_clean_lines,
            "max_clean_lines_inspected": max_clean_lines,
            "accuracy_results": accuracy_results,
            "top_10_correct_idx": correct_idx,
            "top_10_incorrect_idx": incorrect_idx,
        }


def accumulate_results(result, total_recall_results, total_accuracy_results, total_min_clean_lines, total_max_clean_lines):
    """
    累加结果到全局统计。
    """
    for i, recall in enumerate(result["recall_results"]):
        total_recall_results[i] += recall

    for i, accuracy in enumerate(result["accuracy_results"]):
        total_accuracy_results[i] += accuracy

    total_min_clean_lines.append(result["min_clean_lines_inspected"])
    total_max_clean_lines.append(result["max_clean_lines_inspected"])


def compute_global_stats(total_function, total_lines, total_flaw_lines, total_recall_results, total_accuracy_results, total_min_clean_lines, total_max_clean_lines):
    """
    计算全局统计信息。
    """
    avg_recall_results = [recall / total_flaw_lines for recall in total_recall_results] if total_flaw_lines > 0 else []
    avg_accuracy_results = [accuracy / total_function for accuracy in total_accuracy_results] if total_function > 0 else []
    avg_min_clean_lines = sum(total_min_clean_lines) / total_function
    avg_max_clean_lines = sum(total_max_clean_lines) / total_function

    return {
        "total_function": total_function,
        "total_lines": total_lines,
        "total_flaw_lines": total_flaw_lines,
        "avg_top_k_recall": avg_recall_results,
        "avg_top_k_accuracy": avg_accuracy_results,
        "avg_min_clean_lines": avg_min_clean_lines,
        "avg_max_clean_lines": avg_max_clean_lines
    }
    

def label_all_lines(all_lines_score, flaw_line_indices):
    """
    为每一行打标签（漏洞或非漏洞）。

    Args:
        all_lines_score (list): 每行的分数。
        flaw_line_indices (list): 漏洞行索引。

    Returns:
        list: 包含每行分数和标签的列表。
    """
    return [
        [score, 1 if idx in flaw_line_indices else 0]
        for idx, score in enumerate(all_lines_score)
    ]