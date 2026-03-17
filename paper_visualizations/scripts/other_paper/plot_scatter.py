# compare two model acc in a scatter plot, color with sequence length
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import matplotlib.font_manager as fm

# 加载 Times New Roman 字体文件
font_path = '/mnt/nas/share/home/zmz/times.ttf'
font_prop = fm.FontProperties(fname=font_path)

# 设置 Matplotlib 全局字体
plt.rcParams['font.family'] = font_prop.get_name()
json_path1 = 'output_dir4/pifold_baseline/valid/valid_pred_prob_pifold_5170.json'
json_path3 = 'output_dir4/s9_pdb_f30/pifold_baseline_15l_30w/valid/valid_pred_prob_VFN5714.json'
json_path2 = 'output_dir/pifold_baseline_15l/valid/valid_pred_prob_vfn5446.json'
with open(json_path1, 'r') as f:
    data1 = json.load(f)

with open(json_path2, 'r') as f:
    data2 = json.load(f)

with open(json_path3, 'r') as f:
    data3 = json.load(f)

acc_list1 = []
acc_list2 = []
acc_list3 = []
loss_list1 = []
loss_list2 = []
loss_list3 = []
seqlen = []
for i in range(len(data1)):
    acc_list1.append(data1[i]['acc'])
    loss_list1.append(data1[i]['loss'])
print(np.median(acc_list1))
for i in range(len(data2)):
    acc_list2.append(data2[i]['acc'])
    loss_list2.append(data2[i]['loss'])
print(np.median(acc_list2))
for i in range(len(data3)):
    acc_list3.append(data3[i]['acc'])
    loss_list3.append(data2[i]['loss'])
    seqlen.append(len(data3[i]['gt']))
print(np.median(acc_list3))
# ['Pifold', 'VFN', 'VFN++']
# compare two model acc in a scatter plot, color with sequence length
# pifold vs vfn
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.set_xlabel('Pifold Accuracy', fontsize=20,fontproperties=font_prop)
ax.set_ylabel('VFN Accuracy', fontsize=20,fontproperties=font_prop)
ax.tick_params(labelsize=16)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels(['0', '0.5', '1'], fontsize=16,fontproperties=font_prop)
ax.set_yticklabels(['0', '0.5', '1'], fontsize=16,fontproperties=font_prop)
scatter = ax.scatter(acc_list1, acc_list2, s=10, c=seqlen, cmap='viridis',alpha=0.9)
# 创建可映射对象
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(seqlen)
cbar = plt.colorbar(mappable)
cbar.set_label('Sequence Length', fontsize=20, fontproperties=font_prop)
# 对角线
ax.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.savefig('pifold_vs_vfnnew.png', dpi=300)

# pifold vs vfn++

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.set_xlabel('Pifold Accuracy', fontsize=20,fontproperties=font_prop)
ax.set_ylabel('VFN$^{+}$ Accuracy', fontsize=20,fontproperties=font_prop)
ax.tick_params(labelsize=16)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels(['0', '0.5', '1'], fontsize=16,fontproperties=font_prop)
ax.set_yticklabels(['0', '0.5', '1'], fontsize=16,fontproperties=font_prop)
scatter = ax.scatter(acc_list1, acc_list3, s=10, c=seqlen, cmap='viridis',alpha=0.9)
# 创建可映射对象
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(seqlen)
cbar = plt.colorbar(mappable)
cbar.set_label('Sequence Length', fontsize=20,fontproperties=font_prop)
# 对角线
ax.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.savefig('pifold_vs_vfn++new.png', dpi=300)

# vfn vs vfn++
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.set_xlabel('VFN Accuracy', fontsize=20,fontproperties=font_prop)
ax.set_ylabel('VFN$^{+}$ Accuracy', fontsize=20,fontproperties=font_prop)
ax.tick_params(labelsize=16)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels(['0', '0.5', '1'], fontsize=16,fontproperties=font_prop)
ax.set_yticklabels(['0', '0.5', '1'], fontsize=16,fontproperties=font_prop)
scatter = ax.scatter(acc_list2, acc_list3, s=10, c=seqlen, cmap='viridis',alpha=0.9)
# 创建可映射对象
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(seqlen)
cbar = plt.colorbar(mappable)
cbar.set_label('Sequence Length', fontsize=20,fontproperties=font_prop)
# 对角线
ax.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.savefig('vfn_vs_vfn++new.png', dpi=300)
# 三张图一起zhans



