import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

results_files = [
    '/mnt/nas/share2/home/jxr/code/ProteinMPNN/skempi_scripts/paper_results/rde-network_results.csv',
    '/mnt/nas/share2/home/jxr/code/ProteinMPNN/skempi_scripts/paper_results/prompt-ddg_results.csv',
    '/mnt/nas/share2/home/jxr/code/ProteinMPNN/skempi_scripts/paper_results/mpnn-ddg-boltzmann_results.csv',
]

methods = ['RDE-Network', 'Prompt-DDG', 'MPNN-DDG']
Pearson = [0.6447, 0.6772, 0.7118]
Spearman = [0.5584, 0.5910, 0.6346]

# Custom color palette
custom_palette = [
    (171 / 250, 193 / 250, 158 / 250),
    (187 / 250, 130 / 250, 90 / 250), 
    (85 / 250, 104 / 250, 154 / 250), 
]

face_color = (231 / 250, 231 / 250, 240 / 250)


def draw_scatter(save_dir):
    results = [pd.read_csv(file_path) for file_path in results_files]

    # 设置绘图风格
    plt.rcParams.update({'font.size': 16})  # 增加基本字体大小

    # 定义图形和轴
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    # fig.patch.set_facecolor(face_color)

    for ax in axs:
        ax.set_facecolor(face_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1)

    # 读取每个文件并绘制子图
    for i, df in enumerate(results):
        ddg_pred = df['ddG_pred']
        ddg = df['ddG']

        axs[i].scatter(ddg, ddg_pred, color=custom_palette[i], alpha=0.7, s=5.0)
        axs[i].set_title(methods[i])
        axs[i].set_xlabel('Experimental $\\Delta\\Delta G$')
        axs[i].plot([min(ddg.min(), ddg_pred.min()), max(ddg.max(), ddg_pred.max())], 
                    [min(ddg.min(), ddg_pred.min()), max(ddg.max(), ddg_pred.max())], 
                    color='black', linestyle='--', linewidth=1)
        if i == 0:
            axs[i].set_ylabel('Predicted $\\Delta\\Delta G$')

        legend_text = f'Pearson = {Pearson[i]:.4f}\nSpearman = {Spearman[i]:.4f}'
        axs[i].legend([legend_text], loc='upper left', frameon=False)

    # 调整布局以适应标签
    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(save_dir, 'ddg_scatter_plots_combined.pdf')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    draw_scatter(save_dir='/mnt/nas/share2/home/jxr/code/ProteinMPNN/skempi_scripts/fig')