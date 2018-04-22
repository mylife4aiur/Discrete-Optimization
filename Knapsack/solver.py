#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from numba import jit
from pprint import pprint

Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    d = []
    for ind,i in enumerate(items):
        d.append((i.value/i.weight,i.weight,i.index))
    d = np.array(d)
    d_sort =  sorted(d, key=lambda i: i[0],reverse=True)
    d_sort_ind = [int(i[-1]) for i in d_sort]
    items_sorted = [items[i] for i in d_sort_ind]

    # pprint(capacity)
    # pprint(items)
    total_value = sum([i.value for i in items])
    minWeight = min([i.weight for i in items])
    # output_data = dynamicProgrammingReturnResult(items,item_count,capacity)
    output_data = BBReturnResult(items_sorted,item_count,capacity,total_value,minWeight,d_sort_ind)
    return output_data

@jit
def dynamicProgramming(items,item_count,capacity):
    count_matrix = np.zeros((capacity,item_count),dtype=np.int64)
    for j in range(item_count):
        # j-th item
        if j == 0:
            # if the first item
            if items[j].weight == 0 or capacity>items[j].weight:
                pass
            else:
                count_matrix[items[j].weight - 1:,j] = items[j].value

        else:
            # if not the first column
            for i in range(capacity):
                # i-th capacity
                wi = items[j].weight
                vj = items[j].value
                if i < wi-1:
                    # if the capacity is smaller than the weight
                    count_matrix[i][j] = count_matrix[i][j-1]
                else:
                    # if the capacity is larger than the weight
                    if i-wi < 0:
                        count_matrix[i][j] = max( vj,count_matrix[i][j-1])

                    else:
                        count_matrix[i][j] = max( vj + count_matrix[i-wi][j - 1],count_matrix[i][j-1])
    return count_matrix

def dynamicProgrammingReturnResult(items,item_count,capacity):
    count_matrix = dynamicProgramming(items,item_count,capacity)
    dist = np.zeros(item_count)
    opt = count_matrix[-1][-1]
    c = capacity-1
    for i in range(item_count-1,0,-1):
        if count_matrix[c][i] != count_matrix[c][i-1]:
            # item i is selected
            dist[i] = 1
            if items[i].value == count_matrix[c][i]:
                break
            c-=items[i].weight
    dist = [int(i) for i in dist]
    output_data = str(int(opt)) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, dist))
    return output_data

def improvedDynamicProgramming(items,item_count,capacity):
    count_matrix = np.zeros((capacity,2))
    for j in range(item_count):
        # j-th item
        if j == 0:
            # if the first item
            try:
                count_matrix[items[j].weight - 1:] = items[j].value
            except:
                pass
        else:
            # if not the first column
            for i in range(capacity):
                # i-th capacity
                wi = items[j].weight
                vj = items[j].value
                if i < wi-1:
                    # if the capacity is smaller than the weight
                    count_matrix[i][j] = count_matrix[i][j-1]
                else:
                    # if the capacity is larger than the weight
                    if i-wi < 0:
                        count_matrix[i][j] = max( vj,count_matrix[i][j-1])

                    else:
                        count_matrix[i][j] = max( vj + count_matrix[i-wi][j - 1],count_matrix[i][j-1])
    return count_matrix

class TreeNode(object):
    def __init__(self, value,space,estimate,item_number,choice,drop,pos):
        self.value = value
        self.space = space
        self.estimate = estimate
        self.item_number = item_number
        self.choice = choice
        self.drop = drop
        self.left = None
        self.right = None
        self.pos = pos

@jit
def BB(items,capacity,total_value,minWeight):
    current_number = -1
    # 先构建根节点
    root = TreeNode(0,capacity,total_value,current_number,[],[],[])
    root.estimate =  relaxation(items, [], capacity)
    # 建立一个栈，进行深度优先搜索
    stack = []
    stack.append(root)
    best_solution = []
    con = 1
    optimize_value = -1
    # pc =1
    # pd =1
    while stack:
        # 考虑当前节点
        current_root_parent = stack[-1]
        if not current_root_parent.left:
            # 获得当前需要加进来的 item id
            while 1:
                current_number = current_root_parent.item_number + 1

                # 获得当前节点父亲节点的属性
                pv = current_root_parent.value
                ps = current_root_parent.space

                pc = current_root_parent.choice + [current_number]
                pd = current_root_parent.drop
                pos = current_root_parent.pos + [1]
                pe = current_root_parent.estimate
                # pe = relaxation(items, pd, capacity)
                # 对于左子树，选择当前的 item，因此把当前的 item 序号放入 choice 列表
                # pc.append(current_number)
                # con+=1
                # 构建左孩子
                current_left_node = TreeNode(pv + items[current_number].value,
                                             ps - items[current_number].weight,
                                             pe,
                                             current_number,
                                             pc,
                                             pd,
                                             pos)
                # print(pos)
                current_root_parent.left = current_left_node
                # 如果当前节点的空间小于0，或者当前节点的最优估计值小于现有的最优值，都不再访问其下面的子树，不做入栈处理
                if current_left_node.space < 0 \
                        or current_left_node.estimate < optimize_value+1:

                    break
                elif current_number == len(items) - 1  or current_left_node.space < minWeight:
                    if current_left_node.value > optimize_value:
                        optimize_value = current_left_node.value
                        best_solution = current_left_node.choice
                    break
                else:
                    stack.append(current_left_node)

                    # 如果当前背包内的总价值大于当前最优值
                    if current_left_node.value > optimize_value:
                        optimize_value = current_left_node.value
                        best_solution = current_left_node.choice

                    current_root_parent = current_left_node


        # 左子树遍历完要开始遍历右子树
        current_root_parent = stack.pop()

        current_number = current_root_parent.item_number + 1

        pv = current_root_parent.value
        ps = current_root_parent.space
        pc = current_root_parent.choice
        pd = current_root_parent.drop + [current_number]
        pos = current_root_parent.pos + [0]
        pe = relaxation(items, pd, capacity)
        # 再遍历右孩子，不选当前 item
        # pd.append(current_number)
        # con += 1

        current_right_node = TreeNode(pv,
                                ps,
                                pe,
                                current_number,
                                pc,
                                pd,
                                pos)
        # print(pos)

        current_root_parent.right = current_right_node
        # 如果当前节点的空间小于0，或者当前节点的最优估计值小于现有的最优值，就不看子树
        if current_right_node.space < 0 \
                or current_right_node.estimate < optimize_value+1:
            continue
        elif current_number == len(items)-1 or current_right_node.space < minWeight:
            if current_right_node.value > optimize_value:
                optimize_value = current_right_node.value
                best_solution = current_right_node.choice
            continue
        else:
            stack.append(current_right_node)

            # # 如果当前背包内的总价值大于当前最优值
            # if current_right_node.value > optimize_value:
            #     optimize_value = current_right_node.value
            #     best_solution = current_right_node.choice

    # print(con)
    return optimize_value,best_solution


def BBReturnResult(items,item_count,capacity,total_value,minWeight,d_sort_ind):
    optimize_value, best_solution = BB(items,capacity,total_value,minWeight)
    dist = np.zeros(item_count)
    for i in best_solution:
        dist[d_sort_ind[i]] = 1
    # dist = [int(i) for i in dist]
    dist = list(map(int,dist))
    output_data = str(int(optimize_value)) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, dist))
    return output_data


# def relaxation(items,root,capacity ):
#     density = []
#     drop_list = root.drop
#
#     for ind,i in enumerate(items):
#         density.append((i.value/i.weight,i.weight))
#     density = np.array(density)
#     density[drop_list] = (0,0)
#     density_sort =  sorted(density, key=lambda i: i[0],reverse=True)
#
#     m_cap = 0
#     m_val = 0
#     for i in density_sort:
#         if m_cap + i[1] > capacity:
#             frac = (capacity - m_cap)/i[1]
#             m_val += i[0]*frac
#             break
#         else:
#             m_val += i[0]
#     return m_val




def relaxation(items,drop_list,capacity):
    density = []


    for ind,i in enumerate(items):
        density.append((i.value/i.weight,i.weight))
    density = np.array(density)
    density[drop_list] = (0,0)
    density_sort =  sorted(density, key=lambda i: i[0],reverse=True)

    m_cap = 0
    m_val = 0
    for i in density_sort:
        if i[1] == 0:
            break
        frac = (capacity - m_cap) / i[1]
        m_cap += i[1]
        if m_cap > capacity:
            m_val += i[0]*frac*i[1]
            break
        else:
            m_val += i[0]*i[1]
    return m_val


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        # file_location = '/Users/yanghaoyu/learn/离散优化/knapsack/data/ks_30_0'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

