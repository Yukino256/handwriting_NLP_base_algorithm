from typing import List
from math import *
import random
import numpy as np
def calculate_Euc_dis(a: List, b:List):
    if len(a) != len(b):
        raise ValueError('The data dimension differs !!')
    res = 0
    for i in range(len(a)):
        res += (a[i] - b[i]) ** 2
    # print(res)
    return res

def calculate_average(data: List[List]):
    res = [0] * len(data[0])
    for i in range(len(data)):
        for j in range(len(res)):
            res[j] += data[i][j]
    for i in range(len(data[0])):
        res[i] = round(res[i] / len(data), 3)
    # print(res)
    return res

def is_same(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def k_means(datasets: List[List], k: int):
    if len(datasets) < k:
        print('The K  you set exceeds the cases the datasets own ! ')
        return 
    cur_k_nodes = {}
    cluster_nodes = []
    for i in range(0, k):
        cur_k_nodes[i] = []
        cluster_nodes.append(datasets[-i])
        # print(cluster_nodes[i])
    
    while True:
        for i in range(len(datasets)):
            cur = inf
            target_ind = 0
            for j in range(len(cluster_nodes)):
                tmp = calculate_Euc_dis(datasets[i], cluster_nodes[j])
                # print(tmp)
                if cur > tmp:
                    cur = tmp
                    target_ind = j
            cur_k_nodes[target_ind].append(datasets[i])
        # print(cur_k_nodes)
        unchanged_cnt = 0
        for i in range(k):
            new_node = calculate_average(cur_k_nodes[i])
            if is_same(cluster_nodes[i], new_node):
                unchanged_cnt += 1
            cluster_nodes[i] = new_node
        # print(cluster_nodes)
        # print(unchanged_cnt, k)
        if unchanged_cnt == k:
            return [cluster_nodes[i] for i in range(k)], [cur_k_nodes[i] for i in range(k)]
        cur_k_nodes = {key: [] for key in cur_k_nodes}

test = [[random.randint(1, 10) for _ in range(5)] for _ in range(10)]
a, b = k_means(test, 2)
# print(a)
for item in b:
    print(item)