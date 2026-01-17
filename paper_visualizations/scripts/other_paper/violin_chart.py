# plot violin chart
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
json_path1 = 'output_dir4/pifold_baseline/valid/valid_pred_prob_pifold_5170.json'
json_path3 = 'output_dir4/s9_pdb_f30/pifold_baseline_15l_30w/valid/valid_pred_prob_VFN5714.json'
json_path2 = 'output_dir/pifold_baseline_15l/valid/valid_pred_prob_vfn5446.json'
import matplotlib.font_manager as fm

# 加载 Times New Roman 字体文件
font_path = '/mnt/nas/share/home/zmz/times.ttf'
font_prop = fm.FontProperties(fname=font_path)

# 设置 Matplotlib 全局字体
plt.rcParams['font.family'] = font_prop.get_name()
with open(json_path1, 'r') as f:
    data1 = json.load(f)

with open(json_path2, 'r') as f:
    data2 = json.load(f)

with open(json_path3, 'r') as f:
    data3 = json.load(f)

acc_list1 = []
acc_list2 = []
acc_list3 = []
for i in range(len(data1)):
    acc_list1.append(data1[i]['acc'])
print(np.median(acc_list1))
for i in range(len(data2)):
    acc_list2.append(data2[i]['acc'])
print(np.median(acc_list2))
for i in range(len(data3)):
    acc_list3.append(data3[i]['acc'])
print(np.median(acc_list3))
# plot violin chart in one figure
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylim(0, 1)
ax.set_ylabel('Accuracy', fontsize=20,fontproperties=font_prop)
#ax.set_xlabel('Model', fontsize=20,fontproperties=font_prop)
ax.tick_params(labelsize=16)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['PiFold', 'VFN', 'VFN$^{+}$'], fontsize=20,fontproperties=font_prop)
# 为每个数据集分配不同的颜色
colors = [(226/255, 240/255, 217/255), (222/255, 235/255, 247/255) , (248/255, 203/255, 173/255)]
colors =  [(226/255, 255/255, 217/255), (222/255, 235/255, 255/255) , (255/255, 203/255, 173/255)]
#colors = [(246/255, 210/255, 157/255), (198/255, 219/255, 239/255), (134/255, 188/255, 228/255)]
colors = ['blue', 'green', 'red']
parts = ax.violinplot([acc_list1, acc_list2, acc_list3], showmeans=True, showmedians=True)
for i, pc in enumerate( parts['bodies']):
    pc.set_facecolor(colors[i])
plt.show()
for partname in ['cmeans', 'cmedians', 'cmaxes', 'cmins','cbars']:
    part = parts[partname]
    part.set_edgecolor((0.5, 0.5, 0.5))
    part.set_linewidth(1.5)
# save violin chart
fig.savefig('violin_chart_v7.png', dpi=300, bbox_inches='tight')
