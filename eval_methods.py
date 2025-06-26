# import numpy as np
# import more_itertools as mit
# from spot import SPOT, dSPOT
# from spot_mom import SPOT as SPOT_mom
#
#
# def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
#     """
#     使用给定的 `score`、`threshold`（或给定的 `pred`）和 `label` 计算调整后的预测标签。
#     Args:
#             score (np.ndarray): 异常分数
#             label (np.ndarray): 真实标签
#             threshold (float): 异常分数的阈值。
#                     如果一个点的分数低于阈值，则该点被标记为“异常”。
#             pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
#             calc_latency (bool):
#     Returns:
#             np.ndarray: 预测标签
#
#     Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
#     """
#     if label is None:
#         predict = score > threshold
#         return predict, None
#
#     if pred is None:
#         if len(score) != len(label):
#             raise ValueError("score and label must have the same length")
#         predict = score > threshold
#     else:
#         predict = pred
#
#     actual = label > 0.1
#     anomaly_state = False
#     anomaly_count = 0
#     latency = 0
#
#     for i in range(len(predict)):
#         if any(actual[max(i, 0): i + 1]) and predict[i] and not anomaly_state:
#             anomaly_state = True
#             anomaly_count += 1
#             for j in range(i, 0, -1):
#                 if not actual[j]:
#                     break
#                 else:
#                     if not predict[j]:
#                         predict[j] = True
#                         latency += 1
#         elif not actual[i]:
#             anomaly_state = False
#         if anomaly_state:
#             predict[i] = True
#     if calc_latency:
#         return predict, latency / (anomaly_count + 1e-4)
#     else:
#         return predict
#
#
# def calc_point2point(predict, actual):
#     """
#     通过预测和实际计算 f1 分数。
#     Args:
#             predict (np.ndarray): the predict label
#             actual (np.ndarray): np.ndarray
#     Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
#     """
#     TP = np.sum(predict * actual)
#     TN = np.sum((1 - predict) * (1 - actual))
#     FP = np.sum(predict * (1 - actual))
#     FN = np.sum((1 - predict) * actual)
#     precision = TP / (TP + FP + 0.00001)
#     recall = TP / (TP + FN + 0.00001)
#     f1 = 2 * precision * recall / (precision + recall + 0.00001)
#     return f1, precision, recall, TP, TN, FP, FN
#
#
# def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
#     """
#    在给定的分数上运行 POT 方法。
#     :param init_score (np.ndarray): 获取初始化阈值的数据。
#                     For `OmniAnomaly`, it should be the anomaly score of train set.
#     :param: score (np.ndarray): 运行POT方法的数据。
#                     For `OmniAnomaly`, it should be the anomaly score of test set.
#     :param label (np.ndarray): 分数中真实异常的布尔列表
#     :param q (float): 检测级别 (risk)
#     :param level (float): 与初始阈值 t 相关的概率
#     :return dict: pot 结果字典
#     Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
#     """
#
#     print(f"Running POT with q={q}, level={level}..")
#     s = SPOT(q)  # SPOT object
#     s.fit(init_score, score)
#     s.initialize(level=level, min_extrema=False)  # 校准步骤
#     ret = s.run(dynamic=dynamic, with_alarm=False)
#
#     print(len(ret["alarms"]))
#     print(len(ret["thresholds"]))
#
#     pot_th = np.mean(ret["thresholds"])
#     pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
#     if label is not None:
#         p_t = calc_point2point(pred, label)
#         return {
#             "f1": p_t[0],
#             "precision": p_t[1],
#             "recall": p_t[2],
#             "TP": p_t[3],
#             "TN": p_t[4],
#             "FP": p_t[5],
#             "FN": p_t[6],
#             "threshold": pot_th,
#             "latency": p_latency,
#         }
#     else:
#         return {
#             "threshold": pot_th,
#         }
#
#
# def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
#     """
#     Find the best-f1 score by searching best `threshold` in [`start`, `end`).
#     Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
#     """
#
#     print(f"Finding best f1-score by searching for threshold..")
#     if step_num is None or end is None:
#         end = start
#         step_num = 1
#     search_step, search_range, search_lower_bound = step_num, end - start, start
#     if verbose:
#         print("search range: ", search_lower_bound, search_lower_bound + search_range)
#     threshold = search_lower_bound
#     m = (-1.0, -1.0, -1.0)
#     m_t = 0.0
#     m_l = 0
#     for i in range(search_step):
#         threshold += search_range / float(search_step)
#         target, latency = calc_seq(score, label, threshold)
#         if target[0] > m[0]:
#             m_t = threshold
#             m = target
#             m_l = latency
#         if verbose and i % display_freq == 0:
#             print("cur thr: ", threshold, target, m, m_t)
#
#     return {
#         "f1": m[0],
#         "precision": m[1],
#         "recall": m[2],
#         "TP": m[3],
#         "TN": m[4],
#         "FP": m[5],
#         "FN": m[6],
#         "threshold": m_t,
#         "latency": m_l,
#     }
#
#
# def calc_seq(score, label, threshold):
#     predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
#     return calc_point2point(predict, label), latency
#
#
# def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1):
#     best_epsilon = find_epsilon(train_scores, reg_level)
#     pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)
#     if test_labels is not None:
#         p_t = calc_point2point(pred, test_labels)
#         return {
#             "f1": p_t[0],
#             "precision": p_t[1],
#             "recall": p_t[2],
#             "TP": p_t[3],
#             "TN": p_t[4],
#             "FP": p_t[5],
#             "FN": p_t[6],
#             "threshold": best_epsilon,
#             "latency": p_latency,
#             "reg_level": reg_level,
#         }
#     else:
#         return {"threshold": best_epsilon, "reg_level": reg_level}
#
#
# def find_epsilon(errors, reg_level=1):
#     """
#     Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
#     Code from TelemAnom (https://github.com/khundman/telemanom)
#     """
#     e_s = errors
#     best_epsilon = None
#     max_score = -10000000
#     mean_e_s = np.mean(e_s)
#     sd_e_s = np.std(e_s)
#
#     for z in np.arange(2.5, 12, 0.2):
#         epsilon = mean_e_s + sd_e_s * z
#         pruned_e_s = e_s[e_s < epsilon]
#
#         i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )
#         buffer = np.arange(1, 50)
#         i_anom = np.sort(
#             np.concatenate(
#                 (
#                     i_anom,
#                     np.array([i + buffer for i in i_anom]).flatten(),
#                     np.array([i - buffer for i in i_anom]).flatten(),
#                 )
#             )
#         )
#         i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
#         i_anom = np.sort(np.unique(i_anom))
#
#         if len(i_anom) > 0:
#             groups = [list(group) for group in mit.consecutive_groups(i_anom)]
#             # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
#
#             mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
#             sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
#             if reg_level == 0:
#                 denom = 1
#             elif reg_level == 1:
#                 denom = len(i_anom)
#             elif reg_level == 2:
#                 denom = len(i_anom) ** 2
#
#             score = (mean_perc_decrease + sd_perc_decrease) / denom
#
#             if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
#                 max_score = score
#                 best_epsilon = epsilon
#
#     if best_epsilon is None:
#         best_epsilon = np.max(e_s)
#     return best_epsilon
#
#
# def pot_eval_mom(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
#     """
#    在给定的分数上运行 POT 方法。
#     :param init_score (np.ndarray): 获取初始化阈值的数据。
#                     For `OmniAnomaly`, it should be the anomaly score of train set.
#     :param: score (np.ndarray): 运行POT方法的数据。
#                     For `OmniAnomaly`, it should be the anomaly score of test set.
#     :param label (np.ndarray): 分数中真实异常的布尔列表
#     :param q (float): 检测级别 (risk)
#     :param level (float): 与初始阈值 t 相关的概率
#     :return dict: pot 结果字典
#     Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
#     """
#
#     print(f"Running POT with q={q}, level={level}..")
#     s = SPOT_mom(q)  # SPOT object
#     s.fit(init_score, score)
#     s.initialize(level=level, min_extrema=False)  # 校准步骤
#     ret = s.run(dynamic=dynamic, with_alarm=False)
#
#     print(len(ret["alarms"]))
#     print(len(ret["thresholds"]))
#
#     pot_th = np.mean(ret["thresholds"])
#     pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
#     if label is not None:
#         p_t = calc_point2point(pred, label)
#         return {
#             "f1": p_t[0],
#             "precision": p_t[1],
#             "recall": p_t[2],
#             "TP": p_t[3],
#             "TN": p_t[4],
#             "FP": p_t[5],
#             "FN": p_t[6],
#             "threshold": pot_th,
#             "latency": p_latency,
#         }
#     else:
#         return {
#             "threshold": pot_th,
#         }
#
#
# def get_best_f1(score, label):
#     """
#     :param score: 1-D array, input score, tot_length
#     :param label: 1-D array, standard label for anomaly
#     :return: list for results, threshold
#     """
#
#     assert score.shape == label.shape
#     print('***computing best f1***')
#     search_set = []
#     tot_anomaly = 0
#     for i in range(label.shape[0]):
#         tot_anomaly += (label[i] > 0.5)
#     flag = 0
#     cur_anomaly_len = 0
#     cur_min_anomaly_score = 1e5
#     for i in range(label.shape[0]):
#         if label[i] > 0.5:
#             # here for an anomaly
#             if flag == 1:
#                 cur_anomaly_len += 1
#                 cur_min_anomaly_score = score[i] if score[i] < cur_min_anomaly_score else cur_min_anomaly_score
#             else:
#                 flag = 1
#                 cur_anomaly_len = 1
#                 cur_min_anomaly_score = score[i]
#         else:
#             # here for normal points
#             if flag == 1:
#                 flag = 0
#                 search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
#                 search_set.append((score[i], 1, False))
#             else:
#                 search_set.append((score[i], 1, False))
#     if flag == 1:
#         search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
#     search_set.sort(key=lambda x: x[0])
#     best_f1_res = - 1
#     threshold = 1
#     P = 0
#     TP = 0
#     best_P = 0
#     best_TP = 0
#     for i in range(len(search_set)):
#         P += search_set[i][1]
#         if search_set[i][2]:  # for an anomaly point
#             TP += search_set[i][1]
#         precision = TP / (P + 1e-3)
#         recall = TP / (tot_anomaly + 1e-3)
#         f1 = 2 * precision * recall / (precision + recall + 1e-3)
#         if f1 > best_f1_res:
#             best_f1_res = f1
#             threshold = search_set[i][0]
#             best_P = P
#             best_TP = TP
#
#     print('***  best_f1  ***: ', best_f1_res)
#     print('*** threshold ***: ', threshold)
#     t, th = (best_f1_res,
#              best_TP / (best_P + 1e-3),
#              best_TP / (tot_anomaly + 1e-3),
#              best_TP,
#              score.shape[0] - best_P - tot_anomaly + best_TP,
#              best_P - best_TP,
#              tot_anomaly - best_TP), threshold
#     return {
#         'f1': t[0],
#         'precision': t[1],
#         'recall': t[2],
#         'TP': t[3],
#         'TN': t[4],
#         'FP': t[5],
#         'FN': t[6],
#         'threshold': th
#     }


import numpy as np
import more_itertools as mit
from spot import SPOT, dSPOT
from spot_mom import SPOT as SPOT_mom

def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    使用给定的 `score`、`threshold`（或给定的 `pred`）和 `label` 计算调整后的预测标签。
    Args:
            score (np.ndarray): 异常分数
            label (np.ndarray): 真实标签
            threshold (float): 异常分数的阈值。
                    如果一个点的分数低于阈值，则该点被标记为“异常”。
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: 预测标签

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0): i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual):
    """
    通过预测和实际计算 f1 分数。
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    auc = (TP+TN)/(TP+FP+TN+FN)
    return f1, precision, recall, auc, TP, TN, FP, FN


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
   在给定的分数上运行 POT 方法。
    :param init_score (np.ndarray): 获取初始化阈值的数据。
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): 运行POT方法的数据。
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): 分数中真实异常的布尔列表
    :param q (float): 检测级别 (risk)
    :param level (float): 与初始阈值 t 相关的概率
    :return dict: pot 结果字典
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # 校准步骤
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "auc":p_t[3],
            "TP": p_t[4],
            "TN": p_t[5],
            "FP": p_t[6],
            "FN": p_t[7],
            "threshold": pot_th,
            "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "auc":m[3],
        "TP": m[4],
        "TN": m[5],
        "FP": m[6],
        "FN": m[7],
        "threshold": m_t,
        "latency": m_l,
    }


def calc_seq(score, label, threshold):
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency


def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1):
    best_epsilon = find_epsilon(train_scores, reg_level)
    pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)
    if test_labels is not None:
        p_t = calc_point2point(pred, test_labels)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "auc":p_t[3],
            "TP": p_t[4],
            "TN": p_t[5],
            "FP": p_t[6],
            "FN": p_t[7],
            "threshold": best_epsilon,
            "latency": p_latency,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.2):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def pot_eval_mom(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
   在给定的分数上运行 POT 方法。
    :param init_score (np.ndarray): 获取初始化阈值的数据。
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): 运行POT方法的数据。
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): 分数中真实异常的布尔列表
    :param q (float): 检测级别 (risk)
    :param level (float): 与初始阈值 t 相关的概率
    :return dict: pot 结果字典
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT_mom(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # 校准步骤
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "auc":p_t[3],
            "TP": p_t[4],
            "TN": p_t[5],
            "FP": p_t[6],
            "FN": p_t[7],
            "threshold": pot_th,
            "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }


def get_best_f1(score, label):
    """
    :param score: 1-D array, input score, tot_length
    :param label: 1-D array, standard label for anomaly
    :return: list for results, threshold
    """

    assert score.shape == label.shape
    print('***computing best f1***')
    search_set = []
    tot_anomaly = 0
    for i in range(label.shape[0]):
        tot_anomaly += (label[i] > 0.5)
    flag = 0
    cur_anomaly_len = 0
    cur_min_anomaly_score = 1e5
    for i in range(label.shape[0]):
        if label[i] > 0.5:
            # here for an anomaly
            if flag == 1:
                cur_anomaly_len += 1
                cur_min_anomaly_score = score[i] if score[i] < cur_min_anomaly_score else cur_min_anomaly_score
            else:
                flag = 1
                cur_anomaly_len = 1
                cur_min_anomaly_score = score[i]
        else:
            # here for normal points
            if flag == 1:
                flag = 0
                search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
                search_set.append((score[i], 1, False))
            else:
                search_set.append((score[i], 1, False))
    if flag == 1:
        search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
    search_set.sort(key=lambda x: x[0])
    best_f1_res = - 1
    threshold = 1
    P = 0
    TP = 0
    best_P = 0
    best_TP = 0
    for i in range(len(search_set)):
        P += search_set[i][1]
        if search_set[i][2]:  # for an anomaly point
            TP += search_set[i][1]
        precision = TP / (P + 1e-3)
        recall = TP / (tot_anomaly + 1e-3)
        f1 = 2 * precision * recall / (precision + recall + 1e-3)
        if f1 > best_f1_res:
            best_f1_res = f1
            threshold = search_set[i][0]
            best_P = P
            best_TP = TP

    print('***  best_f1  ***: ', best_f1_res)
    print('*** threshold ***: ', threshold)
    t, th = (best_f1_res,
             best_TP / (best_P + 1e-3),
             best_TP / (tot_anomaly + 1e-3),
             (best_TP+score.shape[0] - best_P - tot_anomaly + best_TP)/(best_TP+score.shape[0] - best_P - tot_anomaly + best_TP+best_P - best_TP+tot_anomaly - best_TP),
             best_TP,
             score.shape[0] - best_P - tot_anomaly + best_TP,
             best_P - best_TP,
             tot_anomaly - best_TP), threshold
    return {
        'f1': t[0],
        'precision': t[1],
        'recall': t[2],
        'auc':t[3],
        'TP': t[4],
        'TN': t[5],
        'FP': t[6],
        'FN': t[7],
        'threshold': th
    }
