from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean_jax_exp.visualize import MODE_COLORS, MODE_HATCHES, MODE_LABELS, setup_style

ROOT = Path(__file__).resolve().parent
RUNS = {
    'FCN': ROOT/'results/clean_jax_gradient/runs/clean_jax_D512_adamw_w256_d5_s2000_seed42_20260525_114029',
    'FCN-longskip': ROOT/'results/clean_jax_gradient_longskip/runs/longskip_gradient_D512_adamw_w256_d5_s2000_seed42_20260526_202910',
    'Transformer': ROOT/'results/torch_transformer1d_gradient/runs/torch_transformer_gradient_D512_adamw_p8_d128_h1_L5_m512_manual_s2000_seed42_20260527_094812',
}
MODES = ['x', 'v', 'eps']
KINDS = ['gradient', 'momentum', 'update', 'activation', 'residual']


def read_mm(name: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS[name] / 'logs' / 'matrix_metrics.csv')
    df['experiment'] = name
    return df


def layer_label(df: pd.DataFrame, layer: str) -> str:
    sub = df[df['layer'] == layer]
    if 'layer_label' in sub.columns and len(sub):
        return str(sub['layer_label'].iloc[0])
    return layer


def final_rank90_table(df: pd.DataFrame, layers: list[str] | None = None) -> pd.DataFrame:
    sub = df[(df['step'] == 2000) & (df['matrix_kind'].isin(KINDS))]
    if layers is not None:
        sub = sub[sub['layer'].isin(layers)]
    rows = []
    for layer in list(dict.fromkeys(sub['layer'].tolist())):
        row = {'layer': layer, 'label': layer_label(sub, layer)}
        for kind in KINDS:
            for mode in MODES:
                vals = sub[(sub['layer'] == layer) & (sub['matrix_kind'] == kind) & (sub['mode'] == mode)]['rank90']
                row[f'{kind}_{mode}'] = float(vals.iloc[0]) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ordering(row, prefix='momentum'):
    vals = {m: row[f'{prefix}_{m}'] for m in MODES}
    return ' > '.join([k for k, _ in sorted(vals.items(), key=lambda kv: -kv[1])])


def md_table(df: pd.DataFrame, floatfmt='.1f') -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append('nan' if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def savefig(fig, run_dir: Path, name: str) -> Path:
    out = run_dir / 'figures' / f'{name}.png'
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_output_comparison(out_df: pd.DataFrame, run_dir: Path) -> Path:
    setup_style()
    kinds = ['gradient', 'momentum', 'update', 'activation', 'residual']
    exp_names = ['FCN', 'FCN-longskip', 'Transformer']
    fig, axes = plt.subplots(1, len(kinds), figsize=(18, 3.8), sharey=False)
    width = 0.23
    x = np.arange(len(exp_names), dtype=float)
    for ax, kind in zip(axes, kinds):
        sub = out_df[out_df['kind'] == kind]
        for i, mode in enumerate(MODES):
            vals = [float(sub[sub['experiment'] == exp][mode].iloc[0]) for exp in exp_names]
            bars = ax.bar(x + (i - 1) * width, vals, width=width, color=MODE_COLORS[mode], edgecolor='black', linewidth=0.5, hatch=MODE_HATCHES[mode], label=MODE_LABELS[mode], alpha=0.9)
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + max(vals + [1]) * 0.025, f'{h:.0f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(kind, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=20, ha='right')
        ax.grid(alpha=0.25, axis='y')
    axes[0].set_ylabel('Rank90 effective rank')
    axes[-1].legend(frameon=True, facecolor='white', framealpha=0.85, loc='upper right')
    fig.suptitle('Output Projection Effective Rank90 at Step 2000', fontweight='bold', fontsize=18)
    return savefig(fig, run_dir, 'effective_rank90_output_comparison')


def plot_transformer_momentum_all(tr_final: pd.DataFrame, run_dir: Path) -> Path:
    setup_style()
    labels = tr_final['label'].tolist()
    y = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 9.4))
    height = 0.24
    for i, mode in enumerate(MODES):
        vals = tr_final[f'momentum_{mode}'].to_numpy(dtype=float)
        ax.barh(y + (i - 1) * height, vals, height=height, color=MODE_COLORS[mode], edgecolor='black', linewidth=0.45, hatch=MODE_HATCHES[mode], label=MODE_LABELS[mode], alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Rank90 effective rank')
    ax.set_title('Transformer AdamW Momentum Effective Rank90\nAll Tracked Matrices, Step 2000', fontweight='bold')
    ax.grid(alpha=0.25, axis='x')
    ax.legend(frameon=True, facecolor='white', framealpha=0.85, loc='lower right')
    return savefig(fig, run_dir, 'effective_rank90_transformer_momentum_all_matrices')


def category(layer: str) -> str:
    if layer == 'patch_embed': return 'patch_embed'
    if layer == 'output_proj': return 'output_proj'
    if layer.endswith('.qkv'): return 'qkv'
    if layer.endswith('.attn_out'): return 'attn_out'
    if layer.endswith('.mlp0'): return 'mlp0'
    if layer.endswith('.mlp1'): return 'mlp1'
    return 'other'


def plot_category_means(cat_df: pd.DataFrame, run_dir: Path) -> Path:
    setup_style()
    cats = cat_df['category'].tolist()
    x = np.arange(len(cats), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for i, mode in enumerate(MODES):
        vals = cat_df[f'momentum_{mode}_mean'].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, vals, width=width, color=MODE_COLORS[mode], edgecolor='black', linewidth=0.5, hatch=MODE_HATCHES[mode], label=MODE_LABELS[mode], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha='right')
    ax.set_ylabel('Mean rank90 effective rank')
    ax.set_title('Transformer Momentum Rank90 by Matrix Category', fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    ax.legend(frameon=True, facecolor='white', framealpha=0.85, loc='upper right')
    return savefig(fig, run_dir, 'effective_rank90_transformer_category_means')


def main():
    dfs = {name: read_mm(name) for name in RUNS}
    tr = dfs['Transformer']
    fcn = dfs['FCN']
    longskip = dfs['FCN-longskip']
    run_dir = RUNS['Transformer']

    tr_final = final_rank90_table(tr)
    fcn_final = final_rank90_table(fcn)
    long_final = final_rank90_table(longskip)

    tr_momentum = tr_final[['layer', 'label', 'momentum_x', 'momentum_v', 'momentum_eps']].copy()
    tr_momentum['ordering'] = tr_momentum.apply(ordering, axis=1)
    tr_momentum['max_minus_min'] = tr_momentum[['momentum_x', 'momentum_v', 'momentum_eps']].max(axis=1) - tr_momentum[['momentum_x', 'momentum_v', 'momentum_eps']].min(axis=1)

    out_rows = []
    for exp, df in dfs.items():
        sub = df[(df['layer'] == 'output_proj') & (df['step'] == 2000) & (df['matrix_kind'].isin(KINDS))]
        for kind in KINDS:
            row = {'experiment': exp, 'kind': kind}
            for mode in MODES:
                row[mode] = float(sub[(sub['matrix_kind'] == kind) & (sub['mode'] == mode)]['rank90'].iloc[0])
            out_rows.append(row)
    out_df = pd.DataFrame(out_rows)

    analog_rows = []
    analog = [
        ('input / patch embed', 'FCN input_proj', fcn, 'input_proj', 'Transformer patch_embed', tr, 'patch_embed'),
        ('last MLP up', 'FCN block5_mlp0', fcn, 'block5_mlp0', 'Transformer block5_mlp0', tr, 'blocks.4.mlp0'),
        ('output', 'FCN output_proj', fcn, 'output_proj', 'Transformer output_proj', tr, 'output_proj'),
    ]
    for desc, l1, df1, layer1, l2, df2, layer2 in analog:
        for model, label, df, layer in [('FCN', l1, df1, layer1), ('Transformer', l2, df2, layer2)]:
            sub = df[(df['layer'] == layer) & (df['step'] == 2000) & (df['matrix_kind'] == 'momentum')]
            row = {'comparison': desc, 'model': model, 'layer': label}
            for mode in MODES:
                row[mode] = float(sub[sub['mode'] == mode]['rank90'].iloc[0])
            analog_rows.append(row)
    analog_df = pd.DataFrame(analog_rows)

    tr_cat = tr_final.copy()
    tr_cat['category'] = tr_cat['layer'].map(category)
    cat_rows = []
    for cat, g in tr_cat.groupby('category', sort=False):
        row = {'category': cat, 'n': len(g)}
        for mode in MODES:
            row[f'momentum_{mode}_mean'] = g[f'momentum_{mode}'].mean()
            row[f'gradient_{mode}_mean'] = g[f'gradient_{mode}'].mean()
            row[f'residual_{mode}_mean'] = g[f'residual_{mode}'].mean()
        cat_rows.append(row)
    cat_df = pd.DataFrame(cat_rows)

    out_path = run_dir / 'analysis' / 'transformer_vs_fcn_effective_rank90_analysis.md'
    lines = []
    lines += ['# Effective Rank90 Analysis: Transformer vs FCN', '']
    lines += ['In gradient-dynamics experiments, `effective rank` means `rank90`: the number of singular directions needed to explain 90% of matrix energy.', '']
    lines += ['## Output Projection Rank90 at Step 2000', '', md_table(out_df, '.1f'), '']
    lines += ['## Analogous Matrix Momentum Rank90 at Step 2000', '', md_table(analog_df, '.1f'), '']
    lines += ['## Transformer Momentum Rank90: Every Matrix at Step 2000', '', md_table(tr_momentum, '.1f'), '']
    lines += ['## Transformer Category Means at Step 2000', '', md_table(cat_df, '.2f'), '']
    lines += ['## FCN Momentum Rank90: Every Matrix at Step 2000', '']
    fcn_momentum = fcn_final[['layer', 'label', 'momentum_x', 'momentum_v', 'momentum_eps']].copy()
    fcn_momentum['ordering'] = fcn_momentum.apply(ordering, axis=1)
    lines += [md_table(fcn_momentum, '.1f'), '']
    lines += ['## Main Reading', '']
    lines += ['1. FCN output rank90 reproduces the stable-rank story even more starkly: `x` is tiny, while `v/eps` are high-dimensional at the output matrix.']
    lines += ['2. Transformer output rank90 reverses the FCN signature: `x` has the largest output effective rank, while `eps` is smallest at the output projection.']
    lines += ['3. Transformer effective rank is distributed across patch embedding and internal MLP/QKV matrices, not concentrated in the final projection.']
    lines += ['4. `eps` is still slow in loss, but its difficulty is not a last-layer effective-rank explosion. It is distributed and partially compressed by patch/token structure.']
    out_path.write_text('\n'.join(lines), encoding='utf-8')

    # CSVs for reuse.
    out_df.to_csv(run_dir / 'analysis' / 'rank90_output_comparison.csv', index=False)
    analog_df.to_csv(run_dir / 'analysis' / 'rank90_analogous_matrix_comparison.csv', index=False)
    tr_momentum.to_csv(run_dir / 'analysis' / 'rank90_transformer_momentum_all_matrices.csv', index=False)
    cat_df.to_csv(run_dir / 'analysis' / 'rank90_transformer_category_means.csv', index=False)

    paths = [
        plot_output_comparison(out_df, run_dir),
        plot_transformer_momentum_all(tr_momentum, run_dir),
        plot_category_means(cat_df, run_dir),
    ]
    print(out_path)
    for p in paths:
        print(p)
    print('\nOUTPUT')
    print(out_df.to_string(index=False, float_format=lambda x: f'{x:.1f}'))
    print('\nTRANSFORMER MOMENTUM')
    print(tr_momentum.to_string(index=False, float_format=lambda x: f'{x:.1f}'))


if __name__ == '__main__':
    main()
