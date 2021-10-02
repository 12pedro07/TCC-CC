import matplotlib.pyplot as plt
import statistics
import json
import cv2

from pathlib import Path

DATA_PATH = Path("..", "Results", "UNIFESP")

def prep_and_plot(left_data, left_label, right_data, right_label, title, right_xlim=None):
    fig, axs = plt.subplots(1, 2, figsize=(10,7))

    fig.suptitle(title)

    # Total
    left_mean = statistics.mean(left_data)
    left_std_dev = statistics.stdev(left_data)
    left_samples = len(left_data)
    left_textstr = '\n'.join((
        r'$samples=%d$' % (left_samples, ),
        r'$\mu=%.2f$' % (left_mean, ),
        r'$\sigma=%.2f$' % (left_std_dev, )))
    left_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Porcentagem
    pcts_mean = statistics.mean(right_data)
    pcts_std_dev = statistics.stdev(right_data)
    pcts_samples = len(right_data)
    pcts_textstr = '\n'.join((
        r'$samples=%d$' % (pcts_samples, ),
        r'$\mu=%.2f$' % (pcts_mean, ),
        r'$\sigma=%.2f$' % (pcts_std_dev, )))
    pcts_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Plot result
    axs[0].set_title(left_label)
    axs[0].hist(left_data, bins=100, density=True, facecolor='b')
    axs[0].text(0.55, 0.95, left_textstr, transform=axs[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=left_props)

    axs[1].set_title(right_label)
    axs[1].hist(right_data, bins=100, density=True, facecolor='b')
    axs[1].text(0.55, 0.95, pcts_textstr, transform=axs[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=pcts_props)
    if right_xlim is not None: axs[1].set_xlim(*right_xlim)

    fig.savefig(DATA_PATH.parent / (title+".jpg"))

shapes = []
total = {}
pcts = {}

# Itera na pasta de cada recem-nascido
for dir in DATA_PATH.glob('*'):
    try:
        img = cv2.imread(str(dir / 'bbox_crop.png'), 0)
        shapes.append(img.shape)
    except: pass
    try: 
        json_file = open(dir / 'face_data.json', 'r')
    except FileNotFoundError: 
        print(dir.stem, " has no json file... skipping")
        continue
    data = json.load(json_file)
    for key, value in data['mask'].items():
        try: total[key].append(value['pixel_count'])
        except KeyError: total[key] = [value['pixel_count']]

        try: pcts[key].append(value['pixel_count_pct']*100)
        except KeyError: pcts[key] = [value['pixel_count_pct']*100]
    json_file.close()

widths, heights = zip(*shapes)
prep_and_plot(widths, "Largura", heights, "Alturas", "Tamanhos das bboxes")
for idx, (label, values) in enumerate(total.items()):
    prep_and_plot(total[label], '# Pixels 1 na mascara', pcts[label], '% Pixels 1 na mascara', label, (0,100))