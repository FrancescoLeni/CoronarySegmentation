import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

src = Path(r'C:\Users\franc\OneDrive\crops_metrics')
dst = Path(r'C:\Users\franc\OneDrive\plots')


dice_dict = {'crop': [], 'grid': []}  # these are the test mode
for f in os.listdir(src):
    # crop, grid...
    for m in os.listdir(src / f):
        # Unet..
        if 'skip' not in m:
            for mode in os.listdir(src / f / m):
                # _crop, _grid
                if 'last' not in mode:
                    val = np.load(src / f / m / mode / f'dice.npy')
                    if 'grid' in mode:
                        name = f'{f}_{m}'.replace('_AdamW', '')
                        dice_dict['grid'].append((val, name))
                    elif 'crop' in mode:  # crop
                        name = f'{f}_{m}'.replace('_AdamW', '')
                        dice_dict['crop'].append((val, name))
                    else:  # SAM
                        val = np.load(src / f / m / mode / f'dice.npy')
                        name = 'SAM2'
                        dice_dict['crop'].append((val, name))
                        dice_dict['grid'].append((val, name))

dice_dict['crop'] = sorted(dice_dict['crop'], key=lambda x: np.median(x[0]))
dice_dict['grid'] = sorted(dice_dict['grid'], key=lambda x: np.median(x[0]))

boxes_c = list(reversed([x for x, _ in dice_dict['crop']]))
labels_c= list(reversed([n for _, n in dice_dict['crop']]))

plt.figure(figsize=(19.2, 10.8), dpi=100)
plt.boxplot(boxes_c, labels=labels_c)
plt.axhline(y=np.median(boxes_c[0]), color='red', linestyle='--', linewidth=2, label=f'best median = {np.median(boxes_c[0]): .2f}')
plt.title('Performance on crop extracted Test-set', fontsize=20)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(fontsize=14)
plt.savefig(dst / 'crop_boxplot.png')
plt.close()

boxes_g = list(reversed([x for x, _ in dice_dict['grid']]))
labels_g = list(reversed([n for _, n in dice_dict['grid']]))

plt.figure(figsize=(19.2, 10.8), dpi=100)
plt.boxplot(boxes_g, labels=labels_g)
plt.axhline(y=np.median(boxes_g[0]), color='red', linestyle='--', linewidth=2, label=f'best median = {np.median(boxes_g[0]): .2f}')
plt.title('Performance on grid extracted Test-set', fontsize=20)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(fontsize=14)
plt.savefig(dst / 'grid_boxplot.png')
plt.close()

c_vs_g = sorted([(boxes_c[0], 'tested on crop'), (boxes_g[0], 'tested on grid')], reverse=True, key=lambda x: np.median(x[0]))
boxes = [x for x, _ in c_vs_g]
labels = [n for _, n in c_vs_g]

plt.figure(figsize=(19.2, 10.8), dpi=100)
plt.boxplot(boxes, labels=labels)
plt.axhline(y=np.median(boxes[0]), color='red', linestyle='--', linewidth=2, label=f'best median = {np.median(boxes[0]): .2f}')
plt.title('Performance of Unet3D trained on crops on Test-set', fontsize=20)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(fontsize=14)
plt.savefig(dst / 'crop_vs_grid.png')
plt.close()



