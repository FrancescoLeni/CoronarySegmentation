import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

src = Path(r'C:\Users\franc\OneDrive\crops_metrics')


dice_dict = {'crop': [], 'grid': []}  # these are the test mode
for f in os.listdir(src):
    # crop, grid...
    for m in os.listdir(src / f):
        # Unet..
        if 'skip' not in m:
            for mode in os.listdir(src / f / m):
                # _crop, _grid
                if 'last' not in mode:
                    if 'grid' in mode:
                        val = np.load(src / f / m / mode / f'dice.npy')
                        name = f'{f}_{m}'.replace('_AdamW', '')
                        dice_dict['grid'].append((val, name))
                    else:  # crop
                        val = np.load(src / f / m / mode / f'dice.npy')
                        name = f'{f}_{m}'.replace('_AdamW', '')
                        dice_dict['crop'].append((val, name))

dice_dict['crop'] = sorted(dice_dict['crop'], key=lambda x: np.median(x[0]))
dice_dict['grid'] = sorted(dice_dict['grid'], key=lambda x: np.median(x[0]))

boxes = list(reversed([x for x, _ in dice_dict['crop']]))
labels = list(reversed([n for _, n in dice_dict['crop']]))

plt.figure(figsize=(19.2, 10.8), dpi=100)
plt.boxplot(boxes, labels=labels)
plt.tight_layout()
plt.savefig('crop_boxplot.png')
plt.close()

boxes = list(reversed([x for x, _ in dice_dict['grid']]))
labels = list(reversed([n for _, n in dice_dict['grid']]))

plt.figure(figsize=(19.2, 10.8), dpi=100)
plt.boxplot(boxes, labels=labels)
plt.tight_layout()
plt.savefig('grid_boxplot.png')
plt.close()





