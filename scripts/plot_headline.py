"""Draw the README comparison directly from the four frozen T4 results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    fig, ax = plt.subplots(figsize=(9, 4.2), layout='constrained')
    colors = ['#286b91', '#bd643f']
    for i, rectangle in enumerate([3072, 1757]):
        for j, order in enumerate(['arrival', 'sorted']):
            path = ROOT / f'bench/results/batch-order/armB-mixed-rect{rectangle}-C64-order{order}.json'
            data = json.loads(path.read_text())
            assert data['throughput_valid'] and not data['smoke']
            levels = {row['engine']: row for row in data['levels']}
            ratio = levels['engine']['median_tokens_per_second'] / levels['baseline']['median_tokens_per_second']
            pos = i + (j - .5) * .34
            ax.barh(pos, ratio, height=.28, color=colors[j], label=['Arrival-order baseline','Decode-sorted baseline'][j] if i == 0 else None)
            ax.text(ratio + .03, pos, f'{ratio:.2f}×', va='center', fontsize=11)
    ax.axvline(1, color='#555', linestyle='--', linewidth=1)
    ax.set_yticks([0,1], ['Ragged · rect3072','Ragged · rect1757'])
    ax.invert_yaxis()
    ax.set_xlim(0,2.25)
    ax.set_xlabel('Engine throughput / baseline throughput (1 = parity)')
    ax.set_title('Batch order reverses the throughput result', loc='left', weight='bold', pad=16)
    ax.legend(loc='lower right', frameon=False)
    ax.spines[['top','right','left']].set_visible(False)
    fig.text(.01, -.04, 'Tesla T4 · fp16 · C=64 · matched 1.64 GB KV ceiling · frozen measurements, not a new run', fontsize=9, color='#555')
    output = ROOT / 'docs/images/batch-order.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches='tight')


if __name__ == '__main__':
    main()
