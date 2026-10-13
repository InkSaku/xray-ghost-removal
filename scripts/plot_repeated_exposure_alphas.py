"""Render all repeated nominal exposure settings from the existing analysis.

Does not refit models or modify the frozen inputs. Both exports share a stable
stem next to the existing exposure analysis; staging and QA use OS temp space.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ghost-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'outputs/mhinge_cross_pair_v1/exposure_association'
INPUT = DIRECTORY / 'exposure_association.json'
STEM = 'repeated_settings_alpha'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified_groups():
    data = json.loads(INPUT.read_text(encoding='utf-8'))
    provenance = data['source_provenance']
    manifest_path = ROOT/'outputs/mhinge_cross_pair_v1/analysis_results.json'
    if sha(manifest_path) != provenance['manifest_sha256']:
        raise ValueError('Exposure analysis manifest hash mismatch')
    workbook = ROOT/'data/raw/AI修残影例图/拍摄参数记录.xlsx'
    if sha(workbook) != provenance['workbook_sha256']:
        raise ValueError('Exposure workbook hash mismatch')
    config_path = ROOT/'configs/background_frozen_v1.json'
    if sha(config_path) != provenance['frozen_background']['sha256']:
        raise ValueError('Frozen background config hash mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    by_pair = {row['pair']: row for row in data['pair_results']}
    groups = []
    for group in data['exact_repeated_settings']:
        members = [by_pair[pair] for pair in group['pairs']]
        if len(members) != 2:
            raise ValueError('Figure layout expects two observations per setting')
        for member in members:
            for field in ('kv', 'ma', 'exposure_time_ms'):
                if member[field] != group['setting'][field]:
                    raise ValueError('Group exposure settings differ')
            row = next(r for r in manifest['pair_results'] if
                       (r['light_index'], r['dark_index']) ==
                       (member['light_index'], member['dark_index']))
            block = next(r for r in row['block_results'] if r['block_size'] == 16)
            bg = next(r for r in block['background_results'] if r['background_mode'] == 'masked_hybrid_spline')
            fit = next(r for r in bg['mask_results'] if r['mask_mode'] == 'all')
            if not np.isclose(member['alpha'], fit['fit']['alpha_fold_mean'], rtol=1e-12, atol=0):
                raise ValueError('Alpha differs from frozen all-block fit')
            if member['detected'] != bool(fit['detection_gate']['passed']):
                raise ValueError('Detection flag differs from frozen fit')
        alphas = np.array([m['alpha'] for m in members])
        if not np.isfinite(alphas).all() or np.any(alphas <= 0):
            raise ValueError('Log axis requires positive finite fitted alpha')
        ratio = float(alphas.max()/alphas.min())
        if not np.isclose(ratio, group['alpha_max_to_min_ratio'], rtol=1e-12, atol=0):
            raise ValueError('Recorded ratio does not reproduce')
        groups.append((group, members, ratio))
    groups.sort(key=lambda item: tuple(item[0]['setting'][k] for k in ('kv','ma','exposure_time_ms')))
    if len(groups) != 4:
        raise ValueError('Expected all four repeated-setting groups')
    return groups


def sci_label(value):
    exponent = int(np.floor(np.log10(value)))
    return rf'${value / 10**exponent:.2f}\times10^{{{exponent}}}$'


def figure(groups):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ('Arial Unicode MS', 'Noto Sans CJK SC', 'Heiti SC') if f in installed), None)
    if family is None:
        raise RuntimeError('A Chinese font is required for this slide figure')
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'font.size': 13})
    blue, orange, ink = '#246395', '#bf641a', '#172f45'
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 9.2), sharey=True)
    fig.subplots_adjust(left=.082, right=.98, bottom=.29, top=.70, wspace=.20)
    fig.text(.065, .944, '相同曝光设置下，拟合残影系数的差异', fontsize=25, color=ink)
    fig.text(.065, .887, '完整展示同一会话的 4 组重复设置：其中 3 组的 α 最大/最小比超过 30',
             fontsize=15, color='#526270')
    for index, (ax, (group, members, ratio)) in enumerate(zip(axes, groups)):
        setting = group['setting']
        ax.text(0, 1.24, f'({chr(97+index)})  {setting["kv"]:g} kV · {setting["ma"]:g} mA · {setting["exposure_time_ms"]:g} ms',
                transform=ax.transAxes, fontsize=15, color=ink)
        ax.text(0, 1.11, f'α 最大/最小比  {ratio:.1f}', transform=ax.transAxes,
                fontsize=18, color=blue if ratio > 30 else '#526270')
        values = [m['alpha'] for m in members]
        ax.plot([0,1], values, color='#8c9aa5', lw=1.5, zorder=2)
        for x, member in enumerate(members):
            detected = member['detected']
            ax.scatter(x, member['alpha'], s=150, marker='o',
                       facecolors=blue if detected else 'white',
                       edgecolors=blue if detected else orange, linewidths=2.2, zorder=3)
            ax.annotate(sci_label(member['alpha']), (x, member['alpha']),
                        xytext=(0, 17 if x == 0 else -27), textcoords='offset points',
                        ha='center', va='bottom', fontsize=14, color=ink)
        ax.set_yscale('log'); ax.set_ylim(8e-6, 8e-3); ax.set_xlim(-.32, 1.32)
        ax.set_xticks([0,1], labels=[f'L{m["light_index"]} → D{m["dark_index"]}' for m in members], fontsize=14)
        ax.tick_params(axis='x', length=0, pad=12)
        ax.tick_params(axis='y', which='both', labelsize=13)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2,5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y', which='major', color='#d9e0e5', lw=.8)
        ax.grid(axis='y', which='minor', color='#edf0f2', lw=.5)
        for side in ('top','right'): ax.spines[side].set_visible(False)
        for side in ('left','bottom'): ax.spines[side].set_color('#b5c0c9')
        ax.text(.5, -.14, '2 个观测 pair', transform=ax.transAxes, ha='center', fontsize=12, color='#667580')
    axes[0].set_ylabel('拟合残影系数 α（对数坐标）', fontsize=15, labelpad=12)
    legend = [Line2D([], [], marker='o', lw=0, markersize=9, markerfacecolor=blue,
                     markeredgecolor=blue, label='通过既有残影检出门槛'),
              Line2D([], [], marker='o', lw=0, markersize=9, markerfacecolor='white',
                     markeredgecolor=orange, markeredgewidth=1.7, label='未通过检出门槛；保留拟合 α')]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.065,.172),
               frameon=False, ncol=2, fontsize=14, columnspacing=2.5)
    fig.text(.065, .139, 'L / D：前序 light / 后续 dark；α 为冻结 M1 四折空间 OOF 斜率均值，分析域为 16×16 分块均值。',
             fontsize=12, color='#526270')
    fig.text(.065, .098, '边界：前两组比值含未检出样本；同设置不是受控独立复测，数据仍有源信号限幅风险。',
             fontsize=12, color='#914c19')
    fig.text(.065, .057, '这些是模型拟合系数比，不是已确认的物理残影倍率；每组仅两次观测，未绘制独立重复置信区间。',
             fontsize=12, color='#526270')
    fig.text(.065, .022, '来源：exposure_association.json / exact_repeated_settings（2026-09-22）；4 组全部展示，未重新拟合。',
             fontsize=10.5, color='#667580')
    return fig


def main():
    groups = load_verified_groups()
    fig = figure(groups)
    with tempfile.TemporaryDirectory(prefix='repeated-alpha-') as temporary:
        stage = Path(temporary)
        metadata = {'Description': f'Data SHA256 {sha(INPUT)}; script SHA256 {sha(Path(__file__))}; frozen OOF alpha mean; all four exact repeated settings.'}
        fig.savefig(stage/f'{STEM}.png', dpi=320, facecolor='white', metadata=metadata)
        fig.savefig(stage/f'{STEM}.svg', facecolor='white', metadata=metadata)
        plt.close(fig)
        for path in stage.iterdir():
            shutil.move(str(path), DIRECTORY/path.name)
    for group, members, ratio in groups:
        print(group['setting'], [(m['pair'], m['alpha'], m['detected']) for m in members], f'ratio={ratio:.6f}')
    print(DIRECTORY/f'{STEM}.png')
    print(DIRECTORY/f'{STEM}.svg')


if __name__ == '__main__':
    main()
