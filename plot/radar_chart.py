import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FONT_SIZE = 20 
TITLE_SIZE = 16
LEGEND_SIZE = 13
TICK_SIZE = 13

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'font.weight': 'bold',  
    'axes.titleweight': 'bold', 
    'axes.labelweight': 'bold',  
})


categories = [
    'RS', 'NH', 'SD', 'TP',
    'EH', 'VP', 'FI', 'PC',
    'ED', 'FD', 'ND',
    'RD', 'GD', 'AD',
    'OD', 'HD', 'OT',
    'II', 'BE', 'TS', 'UC',
    'BV', 'HE', 'IR', 'RH',
    'HV', 'PI', 'DM', 'RV'
]


categories = categories[:29]  

deepseek = [7, 3, 3, 4, 1, 12, 1, 5, 12, 12, 14, 24, 3, 22, 8, 15, 17, 8, 7, 4, 39, 4, 3, 16, 4, 2, 0, 18, 0]
deepseekR1 = [24, 13, 15, 18, 11, 18, 10, 9, 30, 28, 36, 33, 11, 32, 22, 23, 31, 20, 23, 22, 57, 18, 20, 34, 20, 9, 6, 30, 19]
deepseekV3 = [26, 8, 13, 5, 9, 10, 3, 5, 9, 20, 22, 32, 17, 31, 18, 16, 25, 7, 3, 3, 36, 7, 5, 20, 8, 6, 5, 21, 1]
deepseekR111 = [83, 66, 69, 50, 56, 53, 65, 34, 66, 74, 64, 58, 33, 47, 46, 42, 59, 71, 62, 63, 77, 72, 53, 83, 72, 66, 54, 68, 67]

values = np.array([deepseek, deepseekR1, deepseekV3, deepseekR111])

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values = np.concatenate((values, values[:, :1]), axis=1)
angles += angles[:1]
categories += categories[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

colors = ['#d15c6b', '#feaac2', '#b5c583', '#87c9c3']
labels = ['DeepSeek-V3(Chinese)', 'DeepSeek-R1(Chinese)', 'DeepSeek-V3(English)', 'DeepSeek-R1(English)']


for i, (model_values, color, label) in enumerate(zip(values, colors, labels)):
    ax.plot(angles, model_values, color=color, linewidth=2, label=label)
    ax.fill(angles, model_values, color=color, alpha=0.1)


ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories[:-1], fontsize=TICK_SIZE) 
ax.set_xticklabels(categories[:-1]) 



plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


ax.set_ylim(0, 100)

# ax.tick_params(axis='y', labelsize=TICK_SIZE)


legend = ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
# for text in legend.get_texts():
#     text.set_fontsize(LEGEND_SIZE)



plt.tight_layout()

plt.show()
plt.savefig('./deepseek_noattack_radar.pdf', format='pdf', bbox_inches='tight')