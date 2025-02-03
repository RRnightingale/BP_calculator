import numpy as np
from scipy.optimize import linprog

def optimize_minmax_winrate(winrate):
    """
    输入: 
        winrate - np.array形状(n, m), 博弈矩阵
    输出:
        params - 最优参数分配(n维向量)
        t - 最大化的最小值
    """
    # print(f"开始最优化 {winrate}")
    n, m = winrate.shape
    # 目标函数: 最小化 -t (等价于最大化 t)
    c = np.zeros(n + 1)  # 变量为 [n1, n2, ..., nn, t]
    c[-1] = -1  # 目标函数系数为 -t

    # 等式约束: sum(n_i) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1  # 前n个变量系数为1
    b_eq = np.array([1.0])

    # 不等式约束: 对于每个表达式j，sum(winrate[i,j] * n_i) >= t → -sum(winrate[i,j] * n_i) + t <= 0
    A_ub = np.zeros((m, n + 1))
    for j in range(m):
        A_ub[j, :n] = -winrate[:, j]  # 前n个系数为 -winrate[i][j]
        A_ub[j, -1] = 1               # t的系数为 +1
    b_ub = np.zeros(m)

    # 变量边界: n_i ∈ [0,1], t无限制
    bounds = [(0, 1) for _ in range(n)] + [(0, 1)]

    # 调用线性规划求解器
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not result.success:
        raise ValueError("优化失败: " + result.message)

    # 提取结果
    strategy = result.x[:n]
    winrate = result.x[-1]
    # print(f"优化完成: {strategy}, {winrate}")
    return strategy, winrate

def cal_match_winrate_with_select(deck_winrate, ally_deck, enemy_deck, ally_select=0, enemy_select=0):
    """
    计算指定卡组组合的胜率
    :param deck_winrate: 卡组胜率矩阵
    :param ally_deck: 我方卡组列表
    :param enemy_deck: 敌方卡组列表
    :param ally_select: 我方选择卡组id
    :param enemy_select: 敌方选择卡组id
    :return: 胜率
    """
    # print(f"select开始计算{ally_deck} select {ally_select} vs {enemy_deck} select {enemy_select}")
    ally_win = deck_winrate[ally_select, enemy_select]  # 对局胜率
    # 如果我方获胜，从我方卡组列表移除该卡组，用剩下的继续比赛
    new_ally_deck = ally_deck.copy()
    new_ally_deck.remove(ally_select)  # 移除已选择的卡组
    _, winrate_if_ally_win = cal_match_winrate(deck_winrate, new_ally_deck, enemy_deck)

    # 如果我方失败，从敌方卡组列表移除该卡组，用剩下的继续比赛
    new_enemy_deck = enemy_deck.copy()
    new_enemy_deck.remove(enemy_select)  # 移除已选择的卡组
    _, winrate_if_ally_lose = cal_match_winrate(deck_winrate, ally_deck, new_enemy_deck)

    total_winrate = ally_win * winrate_if_ally_win + (1 - ally_win) * winrate_if_ally_lose
    # print(f"select计算完毕，胜率：{total_winrate}")
    return total_winrate

def cal_match_winrate(deck_winrate, ally_deck, enemy_deck, info=False):
    """
    给定敌我卡组，计算我方选取卡组的最优策略，以及最优策略下的胜率
    """
    enemy_deck_size = len(enemy_deck)
    ally_deck_size = len(ally_deck)

    if enemy_deck_size == 0:
        return [], 0 # 对方所有卡组跑掉，游戏失败
    if ally_deck_size == 0:
        return [], 1 # 我方所有卡组跑掉，游戏胜利

    if info:
        print(f"开始计算{ally_deck} vs {enemy_deck}")
    # 构建博弈矩阵
    select_winrate_matrix = np.zeros((ally_deck_size,enemy_deck_size))
    for i in range(ally_deck_size):
        for j in range(enemy_deck_size):
            ally_select = ally_deck[i]
            enemy_select = enemy_deck[j]
            select_winrate_matrix[i, j] = cal_match_winrate_with_select(deck_winrate, ally_deck, enemy_deck, ally_select, enemy_select)
    if info:
        print(f"构建博弈矩阵完毕，矩阵为\n{select_winrate_matrix}")
    strategy, winrate = optimize_minmax_winrate(select_winrate_matrix)
    if info:
        print(f"match计算完毕，最优策略为{strategy}, 胜率为{winrate}")
    return strategy, winrate


def cal_match_winrate_with_ban(deck_winrate, ally_deck, enemy_deck):
    """
    给定敌我卡组，计算我方ban卡组的最优策略，以及最优策略下的胜率
    """
    print(f"开始计算 ban {ally_deck} vs {enemy_deck}")
    enemy_deck_size = len(enemy_deck)
    ally_deck_size = len(ally_deck)
    # 构建博弈矩阵
    select_winrate_matrix = np.zeros((ally_deck_size,enemy_deck_size))
    for i in range(ally_deck_size):
        for j in range(enemy_deck_size):
            ally_ban = ally_deck[i]
            enemy_ban = enemy_deck[j]
            new_ally_deck = ally_deck.copy()
            new_ally_deck.remove(ally_ban)  # 移除已选择的卡组

            new_enemy_deck = enemy_deck.copy()
            new_enemy_deck.remove(enemy_ban)  # 移除已选择的卡组

            strategy, winrate = cal_match_winrate(deck_winrate, new_ally_deck, new_enemy_deck)
            select_winrate_matrix[i, j] = winrate
    select_winrate_matrix = select_winrate_matrix.T # 矩阵转置，选择的是对方卡组
    print(f"ban的博弈矩阵为\n{select_winrate_matrix}")
    strategy, winrate = optimize_minmax_winrate(select_winrate_matrix)
    print(f"ban 计算完毕，最优策略为{strategy}, 胜率为{winrate}")
    return strategy, winrate

def cal_match_winrate_with_self_ban(deck_winrate, ally_deck, enemy_deck):
    """
    给定敌我卡组，计算我方自ban卡组的最优策略，以及最优策略下的胜率
    """
    print(f"开始计算 self ban {ally_deck} vs {enemy_deck}")
    enemy_deck_size = len(enemy_deck)
    ally_deck_size = len(ally_deck)
    # 构建博弈矩阵
    select_winrate_matrix = np.zeros((ally_deck_size,enemy_deck_size))
    for i in range(ally_deck_size):
        for j in range(enemy_deck_size):
            ally_ban = ally_deck[i]
            enemy_ban = enemy_deck[j]
            new_ally_deck = ally_deck.copy()
            new_ally_deck.remove(ally_ban)  # 移除已选择的卡组

            new_enemy_deck = enemy_deck.copy()
            new_enemy_deck.remove(enemy_ban)  # 移除已选择的卡组

            strategy, winrate = cal_match_winrate(deck_winrate, new_ally_deck, new_enemy_deck)
            select_winrate_matrix[i, j] = winrate
    print(f"self ban的博弈矩阵为\n{select_winrate_matrix}")
    strategy, winrate = optimize_minmax_winrate(select_winrate_matrix)
    print(f"self ban 计算完毕，最优策略为{strategy}, 胜率为{winrate}")
    return strategy, winrate

if __name__ == "__main__":
    
    deck_winrate = np.array([
        [.55, .6, .5],  # 班尼特对各卡组胜率
        [.45, .3, .45],  # 久闲对各卡组胜率
        [.5, .95, .45]  # 螃蟹对各卡组胜率
    ])

    # 征服3
    ally_deck = [0,1,2]
    enemy_deck = [0,1,2]
    strategy, winrate = cal_match_winrate(deck_winrate=deck_winrate, ally_deck=ally_deck, enemy_deck=enemy_deck, info=True)

    # 3ban1
    ally_deck = [0,1,2]
    enemy_deck = [0,1,2]
    strategy, winrate = cal_match_winrate_with_ban(deck_winrate=deck_winrate, ally_deck=ally_deck, enemy_deck=enemy_deck)

    # 3自ban1
    ally_deck = [0,1,2]
    enemy_deck = [0,1,2]
    strategy, winrate = cal_match_winrate_with_self_ban(deck_winrate=deck_winrate, ally_deck=ally_deck, enemy_deck=enemy_deck)
