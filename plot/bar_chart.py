import matplotlib.pyplot as plt
import numpy as np

TITLE_SIZE = 16
XLABEL_SIZE = 14
YLABEL_SIZE = 14
XTICK_SIZE = 12
YTICK_SIZE = 12
LEGEND_SIZE = 15
BAR_VALUE_SIZE = 12


categories = [
    'Regime subversion', 'National harm', 'State division',
    'Terrorism promotion', 'Ethnic hatred', 'Violence promotion',
    'False information', 'Prohibited content',
    'Ethnic discrimination', 'Faith discrimination'
]

doubao_1_5_pro = [14, 56, 20, 8, 12, 20, 50, 9, 14, 20]
hunyuan_turbo = [1, 9, 1, 1, 1, 3, 1, 0, 0, 2]
moonshot_v1 = [87, 52, 51, 31, 38, 33, 54, 26, 73, 73]
qwen_max = [58, 53, 45, 50, 51, 55, 77, 54, 47, 41]
qwq_32b = [97, 87, 86, 91, 89, 89, 98, 73, 98, 98]

fig, ax = plt.subplots(figsize=(17, 6))


x = np.arange(len(categories))
width = 0.15  

rects1 = ax.bar(x - width * 2, doubao_1_5_pro, width, label='Doubao-1.5-pro-32k-250115', color='#499bc0')
rects2 = ax.bar(x - width, hunyuan_turbo, width, label='Hunyuan-turbo-latest', color='#b9d8ab')
rects3 = ax.bar(x , moonshot_v1, width, label='Moonshot-v1-8k', color='#fe7966')
rects4 = ax.bar(x + width, qwen_max, width, label='Qwen-Max', color='#8fa2dc')
rects5 = ax.bar(x + width * 2, qwq_32b, width, label='QwQ-32B', color='#fac7b3')

ax.set_ylabel('ASR (%)', fontsize=YLABEL_SIZE)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=XTICK_SIZE)
ax.set_ylim(0, 110)
ax.set_xlim(-0.6, len(categories) - 0.4)  

ax.tick_params(axis='y', labelsize=YTICK_SIZE)

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',  
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=BAR_VALUE_SIZE)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)
add_labels(rects5)

ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=LEGEND_SIZE)

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
plt.savefig('./chinese_attack_bar.pdf', format='pdf', bbox_inches='tight')