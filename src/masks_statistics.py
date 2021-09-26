import matplotlib.pyplot as plt
import statistics
import json

from pathlib import Path

DATA_PATH = Path("..", "Results", "UNIFESP_Completo")

total = {}
pcts = {}

# Itera na pasta de cada recem-nascido
for dir in DATA_PATH.glob('*'):
    with open(dir / 'face_data.json', 'r') as json_file:
        data = json.load(json_file)
        for key, value in data['mask'].items():
            try: total[key].append(value['pixel_count'])
            except KeyError: total[key] = [value['pixel_count']]

            try: pcts[key].append(value['pixel_count_pct']*100)
            except KeyError: pcts[key] = [value['pixel_count_pct']*100]

for idx, (label, values) in enumerate(total.items()):
    fig, axs = plt.subplots(1, 2, figsize=(10,7))

    fig.suptitle(label)

    # Total
    total_mean = statistics.mean(total[label])
    total_std_dev = statistics.stdev(total[label])
    total_samples = len(total[label])
    total_textstr = '\n'.join((
        r'$samples=%d$' % (total_samples, ),
        r'$\mu=%.2f$' % (total_mean, ),
        r'$\sigma=%.2f$' % (total_std_dev, )))
    total_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Porcentagem
    pcts_mean = statistics.mean(pcts[label])
    pcts_std_dev = statistics.stdev(pcts[label])
    pcts_samples = len(pcts[label])
    pcts_textstr = '\n'.join((
        r'$samples=%d$' % (pcts_samples, ),
        r'$\mu=%.2f$' % (pcts_mean, ),
        r'$\sigma=%.2f$' % (pcts_std_dev, )))
    pcts_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Plot result
    axs[0].set_title('# De pixels 1 na mascara')
    axs[0].hist(total[label], bins=50, density=True, facecolor='b')
    axs[0].text(0.55, 0.95, total_textstr, transform=axs[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=total_props)

    axs[1].set_title('% Pixels 1 na mascara')
    axs[1].hist(pcts[label], bins=50, density=True, facecolor='b')
    axs[1].text(0.55, 0.95, pcts_textstr, transform=axs[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=pcts_props)
    axs[1].set_xlim(0,100)

    fig.savefig(DATA_PATH.parent / (label+".jpg"))