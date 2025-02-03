# 征服BP ui. 支持3种BP模式，动态卡组数量

import gradio as gr
import numpy as np
import traceback
from bp_calculator import (
    cal_match_winrate,
    cal_match_winrate_with_ban,
    cal_match_winrate_with_self_ban
)

# 核心配置
MAX_SIZE = 5
DEFAULT_SIZE = 3
DEFAULT_NAMES = {
    "ally": ["班尼特", "久闲", "螃蟹", "我方卡组4", "我方卡组5"],
    "enemy": ["螃蟹", "双水", "班尼特", "敌方卡组4", "敌方卡组5"]
}
DEFAULT_WINRATE = [
    [0.55, 0.6, 0.5, 0.5, 0.5],
    [0.45, 0.3, 0.45, 0.5, 0.5],
    [0.5, 0.95, 0.45, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5]
]

def get_default_name(prefix, index):
    """智能获取默认卡组名称"""
    try:
        return DEFAULT_NAMES["ally" if prefix=="我方" else "enemy"][index]
    except IndexError:
        return f"{prefix}卡组{index+1}"

def create_initial_matrix():
    """生成初始矩阵数据"""
    return [
        [get_default_name("我方", i)] + DEFAULT_WINRATE[i][:DEFAULT_SIZE]
        for i in range(DEFAULT_SIZE)
    ]

def update_components(ally_size, enemy_size):
    """统一更新所有动态组件"""
    updates = []
    # 更新我方名称
    for i in range(MAX_SIZE):
        updates.append(gr.update(
            value=get_default_name("我方", i) if i < ally_size else "",
            visible=i < ally_size
        ))
    # 更新敌方名称
    for i in range(MAX_SIZE):
        updates.append(gr.update(
            value=get_default_name("敌方", i) if i < enemy_size else "",
            visible=i < enemy_size
        ))
    # 更新矩阵
    updates.append(gr.update(
        row_count=ally_size,
        col_count=enemy_size + 1,
        headers=[""] + [get_default_name("敌方", i) for i in range(enemy_size)],
        value=[
            [get_default_name("我方", i)] + DEFAULT_WINRATE[i][:enemy_size]
            for i in range(ally_size)
        ]
    ))
    return updates

# 核心计算逻辑
def calculate_strategy(ally_size, enemy_size, *inputs):
    try:
        # 解析输入参数
        params = list(inputs)
        ally_names = params[:MAX_SIZE][:ally_size]
        enemy_names = params[MAX_SIZE:2*MAX_SIZE][:enemy_size]
        matrix = params[2*MAX_SIZE]
        mode = params[2*MAX_SIZE+1]

        # 提取胜率矩阵（跳过每行第一列的名称）  
        winrate = matrix.iloc[:ally_size, 1:1+enemy_size].to_numpy(dtype=np.float64)
        
        if np.any(winrate < 0) or np.any(winrate > 1):
            raise ValueError("所有胜率值必须在0到1之间")

        # 执行计算
        decks = list(range(ally_size)), list(range(enemy_size))
        if mode == "征服（无禁用）":
            strat, wr = cal_match_winrate(winrate, *decks)
            output = format_strategy(strat, ally_names, wr)
        elif mode == "征服 ban1":
            strat, wr = cal_match_winrate_with_ban(winrate, *decks)
            output = f"推荐禁用策略：\n{format_ban(strat, enemy_names)}\n预期胜率：{wr:.1%}"
        elif mode == "征服 自ban1":
            strat, wr = cal_match_winrate_with_self_ban(winrate, *decks)
            output = f"推荐自ban策略：\n{format_ban(strat, ally_names)}\n预期胜率：{wr:.1%}"
        else:
            raise ValueError("未知比赛模式")

        return output

    except Exception as e:
        return f"错误详情：{str(e)}\n\n错误追踪：\n{traceback.format_exc()}"

# 结果格式化
def format_strategy(strategy, names, wr):
    return (
        "推荐出战策略：\n" + 
        "\n".join(f"{n}: {p:.1%}" for n, p in zip(names, strategy)) +
        f"\n预期胜率：{wr:.1%}"
    )

def format_ban(strategy, names):
    return "\n".join(f"ban {n}: {p:.1%}" for n, p in zip(names, strategy))

# 界面构建
with gr.Blocks(title="卡组策略分析器") as demo:
    gr.Markdown("# 🃏 卡组策略分析器")
    
    with gr.Row():
        ally_size = gr.Slider(1, MAX_SIZE, DEFAULT_SIZE, step=1, label="我方卡组数量")
        enemy_size = gr.Slider(1, MAX_SIZE, DEFAULT_SIZE, step=1, label="敌方卡组数量")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 我方卡组")
            ally_deck_names = [
                gr.Textbox(label=f"卡组{i+1}", value=get_default_name("我方", i), 
                          visible=(i < DEFAULT_SIZE))
                for i in range(MAX_SIZE)
            ]
        
        with gr.Column():
            gr.Markdown("### 敌方卡组")
            enemy_deck_names = [
                gr.Textbox(label=f"卡组{i+1}", value=get_default_name("敌方", i),
                          visible=(i < DEFAULT_SIZE))
                for i in range(MAX_SIZE)
            ]
    
    matrix = gr.Dataframe(
        headers=["胜率"] + [get_default_name("敌方", i) for i in range(DEFAULT_SIZE)],
        value=create_initial_matrix(),
        row_count=DEFAULT_SIZE,
        col_count=DEFAULT_SIZE+1,
        datatype=["str"] + ["number"]*DEFAULT_SIZE,
        label="胜率矩阵（行：我方，列：敌方）"
    )
    
    with gr.Row():
        mode = gr.Dropdown(
            ["征服（无禁用）", "征服 ban1", "征服 自ban1"],
            value="征服（无禁用）",
            label="比赛模式"
        )
        submit = gr.Button("开始计算", variant="primary")
    
    output = gr.Textbox(label="分析结果", lines=10)

    # 事件绑定
    for slider in [ally_size, enemy_size]:
        slider.change(
            update_components,
            [ally_size, enemy_size],
            [*ally_deck_names, *enemy_deck_names, matrix]
        )
    
    submit.click(
        calculate_strategy,
        [ally_size, enemy_size, *ally_deck_names, *enemy_deck_names, matrix, mode],
        output
    )

if __name__ == "__main__":
    demo.launch()