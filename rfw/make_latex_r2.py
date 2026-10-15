"""Generate the LaTeX bodies of the revised paper's tables from the per-seed
R2 results (../results_r2/raw_*.csv), so that no number is transcribed by hand.

Each cell is formatted as mean +- 95% CI (1.96 sd / sqrt(n)). When a cell also
appeared in the previous (R1) version of the paper, the generator compares the
two and wraps changed or new cells in \\revised{...} (red in the marked copy).
Output: one .tex snippet per table in ../results_r2/latex/.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(HERE, '..', 'results_r2')
OUT = os.path.join(R2, 'latex')
os.makedirs(OUT, exist_ok=True)

_CACHE = {}


def raw(block):
    if block not in _CACHE:
        _CACHE[block] = pd.read_csv(os.path.join(R2, f'raw_{block}.csv'))
    return _CACHE[block]


def cell(block, label, field='BA'):
    g = raw(block)
    g = g[g['label'] == label]
    if len(g) == 0:
        raise KeyError((block, label))
    x = pd.to_numeric(g[field], errors='coerce').to_numpy(float)
    x = x[~np.isnan(x)]
    m = x.mean()
    c = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, c, int((pd.to_numeric(g['BA']) >= 50).sum()), len(g)


def fmt(m, c, bold=False, ci=True):
    s = f'{m:.1f} \\pm {c:.1f}' if ci else f'{m:.1f}'
    return f'$\\mathbf{{{s}}}$' if bold else f'${s}$'


CHANGES = []
CTX = {'table': ''}


def mark(text, old=None, force=False):
    """Wrap in \\revised{} when new (old is None) or when the value changed;
    log changed R1 values for the response letter."""
    if force or old is None:
        return f'\\revised{{{text}}}'
    norm = lambda t: t.replace('\\mathbf', '').replace('{', '').replace('}', '').replace('$', '').replace(' ', '')
    if norm(text) == norm(old):
        return text
    CHANGES.append((CTX['table'], norm(old), norm(text)))
    return f'\\revised{{{text}}}'


def write(name, body):
    with open(os.path.join(OUT, name + '.tex'), 'w') as f:
        f.write(body)
    print('wrote', name)


# ---------------------------------------------------------------------------
# R1 values (as printed in the submitted version) for change-marking
# ---------------------------------------------------------------------------
R1_ALPHA = {  # defense -> [a0.05, a0.1, a0.5, a1.0]
    'fedavg': ['100.0 \\pm 0.1', '100.0 \\pm 0.0', '99.9 \\pm 0.1', '99.3 \\pm 1.1'],
    'median': ['39.2 \\pm 25.2', '27.2 \\pm 24.0', '36.7 \\pm 25.3', '38.2 \\pm 27.0'],
    'trimmed_mean': ['100.0 \\pm 0.0'] * 4,
    'multikrum': ['100.0 \\pm 0.0', '99.3 \\pm 1.4', '88.3 \\pm 17.4', '87.7 \\pm 12.4'],
    'bulyan': ['100.0 \\pm 0.0', '69.0 \\pm 28.3', '65.7 \\pm 26.9', '62.3 \\pm 21.2'],
    'alignins': ['61.1 \\pm 31.2', '24.9 \\pm 25.3', '25.2 \\pm 21.6', '20.9 \\pm 25.2'],
    'fltrust': ['38.1 \\pm 16.6', '45.3 \\pm 20.6', '5.7 \\pm 1.9', '2.6 \\pm 1.2'],
    'flame': ['100.0 \\pm 0.0', '100.0 \\pm 0.0', '99.0 \\pm 1.3', '94.8 \\pm 5.7'],
    'rfw': ['100.0 \\pm 0.0'] * 4,
    'rfw_pgm': ['13.5 \\pm 18.9', '14.3 \\pm 18.9', '1.5 \\pm 1.1', '1.4 \\pm 1.5'],
}
R1_MAIN = {  # alpha=0.5: BA, MTA, util, FPR
    'fedavg': ('99.9 \\pm 0.1', '93.1 \\pm 0.7', '---', '---'),
    'median': ('36.7 \\pm 25.3', '93.1 \\pm 0.5', '0.0', '---'),
    'trimmed_mean': ('100.0 \\pm 0.0', '92.6 \\pm 0.6', '0.4', '---'),
    'multikrum': ('88.3 \\pm 17.4', '92.4 \\pm 0.8', '0.6', '---'),
    'bulyan': ('65.7 \\pm 26.9', '92.0 \\pm 0.7', '1.1', '---'),
    'rlr': ('0.0 \\pm 0.0', '45.0 \\pm 5.8', '48.1', '---'),
    'alignins': ('25.2 \\pm 21.6', '94.3 \\pm 0.6', '-1.3', '---'),
    'fltrust': ('5.7 \\pm 1.9', '93.4 \\pm 0.8', '-0.3', '---'),
    'flame': ('99.0 \\pm 1.3', '93.1 \\pm 1.1', '0.0', '---'),
    'rfw': ('100.0 \\pm 0.0', '92.7 \\pm 0.6', '0.4', '17.4'),
    'rfw_pgm': ('1.5 \\pm 1.1', '93.1 \\pm 0.7', '-0.1', '4.1'),
}
LBL = {'fedavg': 'FedAvg', 'median': 'Coord-Median', 'trimmed_mean': 'Trimmed-Mean',
       'multikrum': 'Multi-Krum \\cite{blanchard2017}', 'bulyan': 'Bulyan \\cite{elmhamdi2018}',
       'rlr': 'RLR \\cite{ozdayi2021}', 'alignins': 'AlignIns-style \\cite{xu2025alignins}',
       'alignins_orig': 'AlignIns, original \\cite{xu2025alignins}',
       'fltrust': 'FLTrust \\cite{cao2021fltrust}', 'fltrust_r100': 'FLTrust, 100-sample root',
       'flame': 'FLAME-style \\cite{nguyen2022flame}', 'flame_orig': 'FLAME, original \\cite{nguyen2022flame}',
       'rfw': 'ARA (mean)', 'rfw_pgm': '\\textbf{TGM (ours)}'}
R1_NAME = {'alignins': 'AlignIns \\cite{xu2025alignins}', 'flame': 'FLAME (clustering) \\cite{nguyen2022flame}'}
NEW_ROWS = {'alignins_orig', 'fltrust_r100', 'flame_orig'}


def name(d):
    n = LBL[d]
    if d in NEW_ROWS or d in R1_NAME:
        return f'\\revised{{{n}}}'
    return n


def row_tgm(d):
    return '\\rowcolor{tgmhl}' if d == 'rfw_pgm' else ''


def faithful_label(d, kind, **kw):
    """Map a faithful-baseline cell to its label in raw_faithful_digits."""
    if kind == 'alpha':
        return 'faithful_digits', f"alpha|{d}|alpha={kw['alpha']}"
    if kind == 'eta':
        if kw['eta'] == 0.3:
            return 'faithful_digits', f"alpha|{d}|alpha=0.1"
        return 'faithful_digits', f"eta|{d}|eta={kw['eta']}"
    if kind == 'atk':
        if kw['atk'] == 'A3':
            return 'faithful_digits', f"alpha|{d}|alpha=0.1"
        return 'faithful_digits', f"atk|{d}|{kw['atk']}"
    if kind == 'ctrl':
        return 'faithful_digits', f"ctrl|{d}|alpha={kw['alpha']}"
    raise ValueError(kind)


# ---------------------------------------------------------------------------
def t_main():
    CTX['table'] = 'main'
    fed_mta = cell('alpha_all', 'fedavg|alpha=0.5', 'MTA')[0]
    order = ['fedavg', 'median', 'trimmed_mean', 'multikrum', 'bulyan', 'rlr', 'alignins',
             'alignins_orig', 'fltrust', 'fltrust_r100', 'flame', 'flame_orig', 'rfw', 'rfw_pgm']
    lines = []
    for d in order:
        if d in NEW_ROWS:
            blk, lab = faithful_label(d, 'alpha', alpha=0.5)
        else:
            blk, lab = 'alpha_all', f'{d}|alpha=0.5'
        bm, bc, fails, n = cell(blk, lab)
        mm, mc, _, _ = cell(blk, lab, 'MTA')
        util = '---' if d == 'fedavg' else f'{round(fed_mta - mm, 1) + 0.0:.1f}'
        fpr = '---'
        if d in ('rfw', 'rfw_pgm'):
            fpr = f"{cell(blk, lab, 'FPR')[0]:.1f}"
        old = R1_MAIN.get(d)
        ba = fmt(bm, bc, bold=(d == 'rfw_pgm'))
        cols = [mark(ba, old and f'${old[0]}$'), mark(fmt(mm, mc), old and f'${old[1]}$'),
                mark(f'${util}$' if util != '---' else '---', old and (f'${old[2]}$' if old[2] != '---' else '---')),
                mark(f'${fpr}$' if fpr != '---' else '---', old and (f'${old[3]}$' if old[3] != '---' else '---')),
                f'\\revised{{{fails}/{n}}}']
        lines.append(f"{row_tgm(d)}{name(d)} & " + ' & '.join(cols) + ' \\\\')
    write('tab_main', '\n'.join(lines))


def t_mainx():
    CTX['table'] = 'mainx'
    fed_mta = cell('alpha_all', 'fedavg|alpha=0.05', 'MTA')[0]
    lines = []
    singles = [('fedavg', ('100.0 \\pm 0.1', '89.9 \\pm 0.8', '---', '---')),
               ('median', ('39.2 \\pm 25.2', '84.5 \\pm 3.1', '5.4', '---')),
               ('alignins', ('61.1 \\pm 31.2', '87.7 \\pm 2.6', '2.2', '---')),
               ('alignins_orig', None),
               ('fltrust', ('38.1 \\pm 16.6', '90.6 \\pm 1.9', '-0.7', '---')),
               ('fltrust_r100', None), ('flame_orig', None)]
    for d, old in singles:
        if d in NEW_ROWS:
            blk, lab = faithful_label(d, 'alpha', alpha=0.05)
        else:
            blk, lab = 'alpha_all', f'{d}|alpha=0.05'
        bm, bc, fails, n = cell(blk, lab)
        mm, mc, _, _ = cell(blk, lab, 'MTA')
        util = '---' if d == 'fedavg' else f'${round(fed_mta - mm, 1) + 0.0:.1f}$'
        cols = [mark(fmt(bm, bc), old and f'${old[0]}$'), mark(fmt(mm, mc), old and f'${old[1]}$'),
                mark(util, old and (f'${old[2]}$' if old[2] != '---' else '---')), '---',
                f'\\revised{{{fails}/{n}}}']
        lines.append(f"{name(d)} & " + ' & '.join(cols) + ' \\\\')
    # grouped row: Multi-Krum / Bulyan / Trimmed-Mean / FLAME-style
    grp = ['multikrum', 'bulyan', 'trimmed_mean', 'flame']
    bas = [cell('alpha_all', f'{d}|alpha=0.05') for d in grp]
    mtas = [cell('alpha_all', f'{d}|alpha=0.05', 'MTA')[0] for d in grp]
    bmin = min(b[0] for b in bas)
    ba_txt = '$100.0 \\pm 0.0$' if bmin >= 99.95 else f'${bmin:.1f}$--$100.0$'
    mt = f'${min(mtas):.0f}$--${max(mtas):.0f}$'
    fails = sum(b[2] for b in bas); n = sum(b[3] for b in bas)
    lines.append('Multi-Krum/Bulyan/T-Mean/\\revised{FLAME-style} & ' + ba_txt + ' & '
                 + mark(mt, '$78$--$86$') + ' & --- & --- & ' + f'\\revised{{{fails}/{n}}} \\\\')
    for d, old in [('rlr', ('6.0 \\pm 11.8', '11.8 \\pm 1.9', '78.1', '---')),
                   ('rfw', ('100.0 \\pm 0.0', '84.5 \\pm 2.9', '5.5', '20.7')),
                   ('rfw_pgm', ('13.5 \\pm 18.9', '82.8 \\pm 3.6', '7.1', '6.6'))]:
        lab = f'{d}|alpha=0.05'
        bm, bc, fails, n = cell('alpha_all', lab)
        mm, mc, _, _ = cell('alpha_all', lab, 'MTA')
        fpr = f"${cell('alpha_all', lab, 'FPR')[0]:.1f}$" if d in ('rfw', 'rfw_pgm') else '---'
        nm = 'RLR$^\\dagger$ \\cite{ozdayi2021}' if d == 'rlr' else name(d)
        cols = [mark(fmt(bm, bc, bold=(d == 'rfw_pgm')), f'${old[0]}$'), mark(fmt(mm, mc), f'${old[1]}$'),
                mark(f'${fed_mta - mm:.1f}$', f'${old[2]}$'),
                mark(fpr, f'${old[3]}$' if old[3] != '---' else '---'), f'\\revised{{{fails}/{n}}}']
        lines.append(f"{row_tgm(d)}{nm} & " + ' & '.join(cols) + ' \\\\')
    write('tab_mainx', '\n'.join(lines))


def t_alpha():
    CTX['table'] = 'alpha'
    alphas = [0.05, 0.1, 0.5, 1.0]
    order = ['fedavg', 'median', 'trimmed_mean', 'multikrum', 'bulyan', 'rlr', 'alignins',
             'alignins_orig', 'fltrust', 'fltrust_r100', 'flame', 'flame_orig', 'rfw', 'rfw_pgm']
    vals = {}
    for d in order:
        for a in alphas:
            if d in NEW_ROWS:
                blk, lab = faithful_label(d, 'alpha', alpha=a)
            else:
                blk, lab = 'alpha_all', f'{d}|alpha={a}'
            vals[(d, a)] = cell(blk, lab)
    best = {}
    for a in alphas:  # best usable: exclude RLR (destroys MTA)
        cands = [(vals[(d, a)][0], d) for d in order if d not in ('rlr',)]
        best[a] = min(cands)[1]
    lines = []
    for d in order:
        cols = []
        for i, a in enumerate(alphas):
            m, c, f, n = vals[(d, a)]
            if d == 'rlr':
                txt = f'\\textit{{{m:.1f}}}$^\\dagger$'
                old = None
                cols.append(mark(txt, '\\textit{' + {0.05: '6.0', 0.1: '10.0', 0.5: '0.0', 1.0: '0.0'}[a] + '}$^\\dagger$'))
                continue
            txt = fmt(m, c, bold=(best[a] == d))
            old = R1_ALPHA.get(d)
            oldtxt = None if old is None else f'${old[i]}$'
            cols.append(mark(txt, oldtxt))
        lines.append(f"{row_tgm(d)}{name(d)} & " + ' & '.join(cols) + ' \\\\')
    write('tab_alpha', '\n'.join(lines))
    return vals


def t_failure():
    defs = [('median', 'Coord-Median'), ('median_clip', 'Coord-Median $+$ clipping'),
            ('bulyan', 'Bulyan \\cite{elmhamdi2018}'), ('alignins', 'AlignIns-style \\cite{xu2025alignins}'),
            ('fltrust', 'FLTrust \\cite{cao2021fltrust}'), ('rfw_pgm|lam=1', 'TGM, HAS-only ($\\lambda=1$)'),
            ('rfw_pgm', '\\textbf{TGM (ours)}')]
    lines = []
    for key, lab in defs:
        cols = []
        for e in [0.2, 0.3]:
            for a in [0.05, 0.1, 0.5]:
                m, c, f, n = cell('failure50', f'{key}|alpha={a}|eta={e}')
                cols.append(f'{f}/{n}')
        pre = '\\rowcolor{tgmhl}' if key == 'rfw_pgm' else ''
        lines.append(f'{pre}{lab} & ' + ' & '.join(cols) + ' \\\\')
    write('tab_failure', '\n'.join(lines))


def _row(d, cells):
    return f"{row_tgm(d)}{name(d)} & " + ' & '.join(cells) + ' \\\\'


def t_participation():
    CTX['table'] = 'participation'
    R1 = {'fedavg': ['100.0 \\pm 0.0'] * 4,
          'median': ['100.0 \\pm 0.0', '84.1 \\pm 31.2', '8.2 \\pm 4.4', '8.0 \\pm 4.8'],
          'alignins': ['100.0 \\pm 0.0', '100.0 \\pm 0.0', '9.3 \\pm 12.5', '6.5 \\pm 12.2'],
          'fltrust': ['60.0 \\pm 24.4', '40.5 \\pm 23.2', '36.1 \\pm 28.4', '37.0 \\pm 26.7'],
          'rfw': ['100.0 \\pm 0.0'] * 4,
          'rfw_pgm': ['60.6 \\pm 47.3', '65.7 \\pm 41.1', '\\mathbf{4.4 \\pm 3.2}', '\\mathbf{3.2 \\pm 2.6}']}
    lines = []
    for d in ['fedavg', 'median', 'alignins', 'fltrust', 'rfw', 'rfw_pgm']:
        cells = []
        for i, rho in enumerate([0.2, 0.3, 0.4, 0.6]):
            m, c, f, n = cell('participation', f'{d}|rho={rho}')
            cells.append(mark(fmt(m, c), f'${R1[d][i]}$'))
        lines.append(_row(d, cells))
    write('tab_participation', '\n'.join(lines))


def t_eta():
    CTX['table'] = 'eta'
    R1 = {'median': ['1.5 \\pm 0.9', '1.7 \\pm 1.1', '3.3 \\pm 1.2', '27.2 \\pm 24.0', '100.0 \\pm 0.0'],
          'trimmed_mean': ['1.0 \\pm 0.6', '2.4 \\pm 1.4', '99.5 \\pm 0.6', '100.0 \\pm 0.0', '100.0 \\pm 0.0'],
          'multikrum': ['1.0 \\pm 1.2', '11.7 \\pm 16.2', '52.9 \\pm 25.2', '99.3 \\pm 1.4', '100.0 \\pm 0.0'],
          'bulyan': ['0.8 \\pm 0.7', '1.4 \\pm 0.8', '2.9 \\pm 1.4', '69.0 \\pm 28.3', '100.0 \\pm 0.0'],
          'rlr': ['2.8 \\pm 4.7', '12.6 \\pm 14.5', '5.3 \\pm 10.0', '10.0 \\pm 19.6', '0.0 \\pm 0.0'],
          'alignins': ['1.9 \\pm 1.8', '3.9 \\pm 3.4', '1.2 \\pm 1.9', '24.9 \\pm 25.3', '100.0 \\pm 0.0'],
          'fltrust': ['0.5 \\pm 0.4', '2.6 \\pm 3.7', '15.8 \\pm 15.8', '45.3 \\pm 20.6', '68.4 \\pm 21.6'],
          'flame': ['4.7 \\pm 2.9', '56.7 \\pm 10.8', '99.0 \\pm 1.8', '100.0 \\pm 0.0', '100.0 \\pm 0.0'],
          'rfw': ['1.8 \\pm 1.3', '19.5 \\pm 10.4', '95.2 \\pm 9.3', '100.0 \\pm 0.0', '100.0 \\pm 0.0'],
          'rfw_pgm': ['1.5 \\pm 0.8', '1.9 \\pm 1.3', '\\mathbf{2.5 \\pm 1.2}', '\\mathbf{14.3 \\pm 18.9}', '100.0 \\pm 0.0']}
    etas = [0.05, 0.1, 0.2, 0.3, 0.4]
    lines = []
    for d in ['median', 'trimmed_mean', 'multikrum', 'bulyan', 'rlr', 'alignins', 'alignins_orig',
              'fltrust', 'flame', 'flame_orig', 'rfw', 'rfw_pgm']:
        cells = []
        for i, e in enumerate(etas):
            if d in NEW_ROWS:
                blk, lab = faithful_label(d, 'eta', eta=e)
            else:
                blk, lab = 'eta_all', f'{d}|eta={e}'
            m, c, f, n = cell(blk, lab)
            bold = d == 'rfw_pgm' and e in (0.2, 0.3)
            old = R1.get(d)
            cells.append(mark(fmt(m, c, bold=bold), None if old is None else f'${old[i]}$'))
        nm = 'RLR$^\\dagger$ \\cite{ozdayi2021}' if d == 'rlr' else name(d)
        lines.append(f"{row_tgm(d)}{nm} & " + ' & '.join(cells) + ' \\\\')
    write('tab_eta', '\n'.join(lines))


def t_attacks():
    CTX['table'] = 'attacks'
    R1 = {'fedavg': ['88.8 \\pm 19.3', '99.6 \\pm 0.5', '100.0 \\pm 0.0', '97.9 \\pm 1.5'],
          'median': ['30.9 \\pm 17.9', '18.1 \\pm 7.7', '27.2 \\pm 24.0', '6.9 \\pm 3.0'],
          'trimmed_mean': ['100.0 \\pm 0.0', '92.2 \\pm 4.6', '100.0 \\pm 0.0', '55.1 \\pm 10.2'],
          'multikrum': ['35.7 \\pm 22.0', '100.0 \\pm 0.0', '99.3 \\pm 1.4', '99.5 \\pm 0.4'],
          'bulyan': ['1.5 \\pm 0.9', '7.9 \\pm 3.6', '69.0 \\pm 28.3', '54.3 \\pm 19.3'],
          'alignins': ['45.8 \\pm 26.1', '99.9 \\pm 0.2', '24.9 \\pm 25.3', '97.8 \\pm 0.9'],
          'fltrust': ['\\mathbf{1.7 \\pm 1.0}', '\\mathbf{1.7 \\pm 1.0}', '45.3 \\pm 20.6', '\\mathbf{5.7 \\pm 3.5}'],
          'flame': ['100.0 \\pm 0.0', '100.0 \\pm 0.1', '100.0 \\pm 0.0', '98.9 \\pm 0.7'],
          'rfw': ['77.9 \\pm 10.0', '100.0 \\pm 0.1', '100.0 \\pm 0.0', '98.7 \\pm 0.7'],
          'rfw_pgm': ['2.5 \\pm 1.5', '17.0 \\pm 7.1', '\\mathbf{14.3 \\pm 18.9}', '7.9 \\pm 3.5']}
    atks = ['A1', 'A2', 'A3', 'adaptive', 'adaptive_opt']
    order = ['fedavg', 'median', 'trimmed_mean', 'multikrum', 'bulyan', 'alignins', 'alignins_orig',
             'fltrust', 'flame', 'flame_orig', 'rfw', 'rfw_pgm']
    vals = {}
    for d in order:
        for a in atks:
            if d in NEW_ROWS:
                blk, lab = faithful_label(d, 'atk', atk=a)
            else:
                blk, lab = 'variants_all', f'{a}|{d}'
            vals[(d, a)] = cell(blk, lab)
    best = {a: min((vals[(d, a)][0], d) for d in order if d != 'fedavg')[1] for a in atks}
    lines = []
    for d in order:
        cells = []
        for i, a in enumerate(atks):
            m, c, f, n = vals[(d, a)]
            txt = fmt(m, c, bold=(best[a] == d))
            old = R1.get(d)
            cells.append(mark(txt, None if (old is None or i >= 4) else f'${old[i]}$'))
        lines.append(_row(d, cells))
    write('tab_attacks', '\n'.join(lines))


def t_adaptive():
    CTX['table'] = 'adaptive'
    lines = []
    for a in [0.1, 0.5]:
        for e in [0.1, 0.2, 0.3]:
            def g(atk, d):
                return cell('adaptive', f'{atk}|{d}|alpha={a}|eta={e}')
            def pk(d):
                if e == 0.1:
                    return None
                return cell('adaptive_pk', f'adaptive_opt_pk|{d}|alpha={a}|eta={e}')
            t_a3, t_h, t_o, t_p = g('A3', 'rfw_pgm'), g('adaptive', 'rfw_pgm'), g('adaptive_opt', 'rfw_pgm'), pk('rfw_pgm')
            ratio = cell('adaptive', f'adaptive_opt|rfw_pgm|alpha={a}|eta={e}', 'trust_ratio')[0]
            m_a3, m_o, m_p = g('A3', 'median'), g('adaptive_opt', 'median'), pk('median')
            f_o = g('adaptive_opt', 'fltrust')
            fm = lambda x: '---' if x is None else fmt(x[0], x[1])
            cells = [f'${a}$', f'${e}$', fm(t_a3), fm(t_h), fm(t_o), fm(t_p), f'${ratio:.2f}$',
                     fm(m_a3), fm(m_o), fm(m_p), fm(f_o)]
            lines.append(' & '.join(cells) + ' \\\\')
    write('tab_adaptive', '\n'.join(lines))


def t_fmnist():
    CTX['table'] = 'fmnist'
    order = ['fedavg', 'median', 'trimmed_mean', 'multikrum', 'bulyan', 'rlr', 'alignins',
             'alignins_orig', 'fltrust', 'fltrust_r100', 'flame', 'flame_orig', 'rfw', 'rfw_pgm']
    alphas = [0.1, 0.5, 1.0]
    vals = {}
    for d in order:
        for a in alphas:
            if d in NEW_ROWS:
                blk, lab = 'faithful_fmnist', f'{d}|alpha={a}|eta=0.3'
            else:
                blk, lab = 'fmnist_main', f'{d}|alpha={a}|eta=0.3'
            vals[(d, a)] = (cell(blk, lab), cell(blk, lab, 'MTA'))
    best = {a: min((vals[(d, a)][0][0], d) for d in order if d not in ('rlr', 'fedavg'))[1] for a in alphas}
    lines = []
    for d in order:
        cells = []
        for a in alphas:
            (m, c, f, n), _ = vals[(d, a)]
            if d == 'rlr':
                cells.append(f'\\textit{{{m:.1f}}}$^\\dagger$')
            else:
                cells.append(fmt(m, c, bold=(best[a] == d)) + f' ({f})')
        mm, mc, _, _ = vals[(d, 0.5)][1]
        cells.append(fmt(mm, mc))
        nm = 'RLR$^\\dagger$ \\cite{ozdayi2021}' if d == 'rlr' else LBL[d]
        pre = '\\rowcolor{tgmhl}' if d == 'rfw_pgm' else ''
        lines.append(f'{pre}{nm} & ' + ' & '.join(cells) + ' \\\\')
    write('tab_fmnist', '\n'.join(lines))


def t_fmnist2():
    CTX['table'] = 'fmnist2'
    order = ['median', 'multikrum', 'bulyan', 'alignins', 'alignins_orig', 'fltrust', 'flame',
             'flame_orig', 'rfw', 'rfw_pgm']
    lines = []
    for d in order:
        cells = []
        for e in [0.1, 0.2, 0.4]:
            blk, lab = ('faithful_fmnist', f'{d}|alpha=0.5|eta={e}') if d in NEW_ROWS else ('fmnist_eta', f'{d}|alpha=0.5|eta={e}')
            m, c, f, n = cell(blk, lab)
            cells.append(fmt(m, c))
        for atk in ['A1', 'A2', 'adaptive', 'adaptive_opt']:
            try:
                m, c, f, n = cell('fmnist_attacks', f'{atk}|{d}|alpha=0.5|eta=0.3')
                cells.append(fmt(m, c))
            except KeyError:
                cells.append('---')
        pre = '\\rowcolor{tgmhl}' if d == 'rfw_pgm' else ''
        lines.append(f'{pre}{LBL[d]} & ' + ' & '.join(cells) + ' \\\\')
    write('tab_fmnist2', '\n'.join(lines))


def t_unsw():
    CTX['table'] = 'unsw'
    R1 = {'median': ['23.8 \\pm 12.6', '57.0 \\pm 9.6', '98.6 \\pm 1.0', '99.3 \\pm 1.0', '100.0 \\pm 0.1'],
          'trimmed_mean': ['30.0 \\pm 16.9', '81.5 \\pm 8.1', '99.8 \\pm 0.1', '99.7 \\pm 0.5', '99.9 \\pm 0.1'],
          'multikrum': ['\\mathbf{2.8 \\pm 2.0}', '26.5 \\pm 34.3', '86.5 \\pm 26.2', '99.9 \\pm 0.1', '100.0 \\pm 0.0'],
          'bulyan': ['5.4 \\pm 3.4', '\\mathbf{15.2 \\pm 9.8}', '80.4 \\pm 14.8', '100.0 \\pm 0.1', '100.0 \\pm 0.0'],
          'alignins': ['63.1 \\pm 13.0', '93.0 \\pm 4.8', '80.9 \\pm 36.9', '99.7 \\pm 0.3', '99.9 \\pm 0.1'],
          'fltrust': ['15.4 \\pm 12.5', '28.8 \\pm 23.7', '62.3 \\pm 30.6', '99.3 \\pm 0.5', '97.9 \\pm 4.0'],
          'flame': ['73.1 \\pm 16.6', '98.6 \\pm 0.9', '99.7 \\pm 0.1', '99.7 \\pm 0.3', '100.0 \\pm 0.0'],
          'rfw': ['48.7 \\pm 13.3', '95.6 \\pm 2.6', '99.9 \\pm 0.1', '99.8 \\pm 0.2', '99.9 \\pm 0.1'],
          'rfw_pgm': ['18.2 \\pm 6.0', '28.4 \\pm 12.0', '85.2 \\pm 9.2', '99.3 \\pm 0.8', '99.9 \\pm 0.1']}
    etas = [0.05, 0.1, 0.2, 0.3, 0.4]
    order = ['median', 'trimmed_mean', 'multikrum', 'bulyan', 'alignins', 'alignins_orig', 'fltrust',
             'fltrust_r100', 'flame', 'flame_orig', 'rfw', 'rfw_pgm']
    vals = {}
    for d in order:
        for e in etas:
            blk, lab = ('faithful_unsw', f'{d}|eta={e}') if d in NEW_ROWS else ('unsw', f'{d}|eta={e}')
            vals[(d, e)] = cell(blk, lab)
    best = {e: min((vals[(d, e)][0], d) for d in order)[1] for e in etas}
    lines = []
    for d in order:
        cells = []
        for i, e in enumerate(etas):
            m, c, f, n = vals[(d, e)]
            old = R1.get(d)
            cells.append(mark(fmt(m, c, bold=(best[e] == d and e <= 0.3)), None if old is None else f'${old[i]}$'))
        lines.append(_row(d, cells))
    write('tab_unsw', '\n'.join(lines))


def t_quant():
    CTX['table'] = 'quant'
    R1 = {0.2: {'median': ['2.3 \\pm 0.9', '2.2 \\pm 0.9', '2.9 \\pm 1.4', '7.2 \\pm 5.9'],
                'multikrum': ['70.2 \\pm 34.5', '75.9 \\pm 35.3', '80.9 \\pm 37.3', '10.3 \\pm 6.6'],
                'alignins': ['0.3 \\pm 0.3', '0.3 \\pm 0.3', '0.2 \\pm 0.2', '0.6 \\pm 0.7'],
                'fltrust': ['21.7 \\pm 30.3', '21.4 \\pm 30.0', '22.9 \\pm 30.5', '19.6 \\pm 30.5'],
                'rfw_pgm': ['1.7 \\pm 0.9', '1.6 \\pm 0.8', '3.0 \\pm 1.5', '3.4 \\pm 2.4']},
          0.3: {'median': ['8.3 \\pm 4.4', '8.1 \\pm 4.1', '5.9 \\pm 2.6', '29.0 \\pm 23.6'],
                'multikrum': ['98.6 \\pm 2.7', '98.6 \\pm 2.7', '87.7 \\pm 24.1', '63.5 \\pm 39.3'],
                'alignins': ['8.6 \\pm 11.3', '10.2 \\pm 16.2', '3.2 \\pm 3.7', '1.6 \\pm 1.9'],
                'fltrust': ['36.1 \\pm 28.4', '36.3 \\pm 28.4', '37.4 \\pm 29.1', '36.4 \\pm 32.3'],
                'rfw_pgm': ['4.5 \\pm 3.2', '4.4 \\pm 3.1', '4.0 \\pm 3.3', '19.3 \\pm 15.2']}}
    out = []
    for e in [0.2, 0.3]:
        out.append(f'\\rowcolor{{white}}\\multicolumn{{6}}{{l}}{{\\emph{{$\\eta = {e:.2f}$}}}}\\\\')
        for d in ['median', 'multikrum', 'alignins', 'fltrust', 'rfw_pgm']:
            cells = []
            for i, b in enumerate([0, 16, 8, 4, 2]):
                m, c, f, n = cell('quant', f'{d}|eta={e}|bits={b}')
                old = None if b == 0 else R1[e][d][i - 1]
                cells.append(mark(fmt(m, c), None if old is None else f'${old}$'))
            out.append(_row(d, cells))
        if e == 0.2:
            out.append('\\midrule')
    write('tab_quant', '\n'.join(out))


def t_capacity():
    CTX['table'] = 'capacity'
    R1 = {'median': ['2.4 \\pm 2.8', '9.1 \\pm 5.6', '22.3 \\pm 11.5', '100.0 \\pm 0.0'],
          'multikrum': ['96.8 \\pm 4.0', '100.0 \\pm 0.0', '81.3 \\pm 36.7', '100.0 \\pm 0.0'],
          'bulyan': ['1.1 \\pm 1.5', '6.7 \\pm 4.1', '60.4 \\pm 47.5', '100.0 \\pm 0.0'],
          'alignins': ['34.0 \\pm 31.9', '85.5 \\pm 26.3', '78.6 \\pm 37.7', '100.0 \\pm 0.0'],
          'fltrust': ['9.4 \\pm 17.9', '24.7 \\pm 28.4', '25.2 \\pm 20.2', '69.5 \\pm 33.3'],
          'rfw_pgm': ['\\mathbf{2.1 \\pm 2.4}', '\\mathbf{4.2 \\pm 3.4}', '\\mathbf{14.0 \\pm 12.9}', '100.0 \\pm 0.0']}
    etas = [0.1, 0.2, 0.3, 0.4]
    order = ['median', 'multikrum', 'bulyan', 'alignins', 'fltrust', 'rfw_pgm']
    vals = {(d, e): cell('capacity', f'{d}|eta={e}') for d in order for e in etas}
    best = {e: min((vals[(d, e)][0], d) for d in order)[1] for e in etas}
    lines = []
    for d in order:
        cells = []
        for i, e in enumerate(etas):
            m, c, f, n = vals[(d, e)]
            cells.append(mark(fmt(m, c, bold=(best[e] == d)), f'${R1[d][i]}$'))
        lines.append(_row(d, cells))
    write('tab_capacity', '\n'.join(lines))


def t_intermittent():
    CTX['table'] = 'intermittent'
    R1 = {'TGM HAS-only (lam=1)': ['1.2 \\pm 0.3', '1.6 \\pm 0.5', '1.2 \\pm 0.6'],
          'TGM fusion (lam=0.5)': ['1.2 \\pm 0.3', '1.8 \\pm 0.6', '1.3 \\pm 0.6'],
          'TGM prov-only (lam=0)': ['1.2 \\pm 0.3', '1.9 \\pm 0.7', '1.7 \\pm 1.0'],
          'Coord-Median': ['0.9 \\pm 0.3', '1.6 \\pm 0.7', '1.4 \\pm 0.7'],
          'AlignIns': ['2.6 \\pm 1.3', '2.3 \\pm 2.1', '0.7 \\pm 0.4']}
    names = {'TGM HAS-only (lam=1)': 'TGM HAS-only ($\\lambda=1$)',
             'TGM fusion (lam=0.5)': 'TGM fusion ($\\lambda=0.5$)',
             'TGM prov-only (lam=0)': 'TGM provenance-only ($\\lambda=0$)',
             'Coord-Median': 'Coord-Median', 'AlignIns': '\\revised{AlignIns-style} \\cite{xu2025alignins}'}
    lines = []
    for k in R1:
        cells = []
        for i, e in enumerate([0.1, 0.2, 0.3]):
            m, c, f, n = cell('intermittent', f'{k}|eta={e}')
            cells.append(mark(fmt(m, c), f'${R1[k][i]}$'))
        pre = '\\rowcolor{tgmhl}' if k == 'TGM fusion (lam=0.5)' else ''
        lines.append(f'{pre}{names[k]} & ' + ' & '.join(cells) + ' \\\\')
    write('tab_intermittent', '\n'.join(lines))


def t_warmup():
    CTX['table'] = 'warmup'
    order = [('median', 'Coord-Median'), ('bulyan', 'Bulyan \\cite{elmhamdi2018}'),
             ('fltrust', 'FLTrust \\cite{cao2021fltrust}'), ('alignins', 'AlignIns-style \\cite{xu2025alignins}'),
             ('rfw_pgm', '\\textbf{TGM} (warm-up $\\hat h$)'), ('rfw_pgm_online', '\\textbf{TGM} (per-round $\\hat h$)')]
    lines = []
    for d, nm in order:
        cells = []
        for a in [0.5, 0.1]:
            for tag in ['clean', 'contam', 'nowarm']:
                m, c, f, n = cell('warmup', f'{tag}|{d}|alpha={a}|eta=0.3')
                cells.append(fmt(m, c))
        pre = '\\rowcolor{tgmhl}' if d.startswith('rfw') else ''
        lines.append(f'{pre}{nm} & ' + ' & '.join(cells) + ' \\\\')
    write('tab_warmup', '\n'.join(lines))


def t_ablation():
    CTX['table'] = 'ablation'
    rows = [('FedAvg (no defense)', 'FedAvg', 'fedavg|alpha=0.5', ('100.0 \\pm 0.0', '100.0 \\pm 0.1')),
            ('Coord-Median (midpoint, no clipping)', 'Coord-Median (midpoint, no clip)', 'median|alpha=0.5', ('27.2 \\pm 24.0', '39.2 \\pm 25.2')),
            ('Coord-Median $+$ clipping', 'Coord-Median + clipping', None, None),
            ('Uniform weighted median, no clipping', 'Uniform weighted median, no clip', None, None),
            ('TGM $-$ trust (uniform weighted median $+$ clipping)', 'TGM - trust (uniform weighted median + clip)', None, ('29.1 \\pm 23.6', '46.1 \\pm 28.5')),
            ('TGM $-$ median ($=$ ARA mean)', 'TGM - median (trust-weighted mean, ARA)', None, ('100.0 \\pm 0.0', '100.0 \\pm 0.0')),
            ('TGM w/ geometric median (not coordinate)', 'TGM geometric median', None, ('100.0 \\pm 0.0', '100.0 \\pm 0.0')),
            ('TGM, HAS-only trust ($\\lambda=1$)', 'TGM HAS-only (lam=1)', None, ('23.5 \\pm 25.1', '23.5 \\pm 25.1')),
            ('TGM, provenance-only trust ($\\lambda=0$)', 'TGM provenance-only (lam=0)', None, ('14.9 \\pm 18.8', '\\mathbf{5.1 \\pm 3.6}')),
            ('\\textbf{TGM full ($\\lambda=0.5$) (ours)}', 'TGM full (lam=0.5)', None, ('\\mathbf{14.3 \\pm 18.9}', '13.5 \\pm 18.9'))]
    new_names = {'Coord-Median (midpoint, no clipping)', 'Coord-Median $+$ clipping',
                 'Uniform weighted median, no clipping', 'TGM $-$ trust (uniform weighted median $+$ clipping)'}
    fm_map = {'FedAvg': ('fmnist_main', 'fedavg|alpha=0.5|eta=0.3'),
              'Coord-Median (midpoint, no clip)': ('fmnist_main', 'median|alpha=0.5|eta=0.3'),
              'TGM full (lam=0.5)': ('fmnist_main', 'rfw_pgm|alpha=0.5|eta=0.3')}
    lines = []
    for nm, key, _, old in rows:
        cells = []
        for j, a in enumerate([0.1, 0.05, 0.5]):
            m, c, f, n = cell('ablation', f'{key}|alpha={a}')
            o = None if (old is None or j == 2) else f'${old[j]}$'
            cells.append(mark(fmt(m, c), o))
        blk, lab = fm_map.get(key, ('fmnist_ablation', f'{key}|alpha=0.5'))
        m, c, f, n = cell(blk, lab)
        cells.append(mark(fmt(m, c)))
        label = f'\\revised{{{nm}}}' if nm in new_names else nm
        pre = '\\rowcolor{tgmhl}' if key == 'TGM full (lam=0.5)' else ''
        lines.append(f'{pre}{label} & ' + ' & '.join(cells) + ' \\\\')
    write('tab_ablation', '\n'.join(lines))


def t_score():
    CTX['table'] = 'score'
    rows = [('Full score: direction $+$ norm, $s=r/(1{+}r)$', 'TGM full (lam=0.5)'),
            ('Direction term only', 'score: direction term only'),
            ('Norm term only', 'score: norm term only'),
            ('Cosine distance to the mean $+$ norm', 'score: cosine distance + norm'),
            ('Bounded map $s=1-e^{-r}$', 'squash: 1-exp(-r)'),
            ('No bounded map, $s=r$', 'squash: none (s=r)')]
    lines = []
    for nm, key in rows:
        cells = []
        for a in [0.05, 0.1, 0.5]:
            m, c, f, n = cell('ablation', f'{key}|alpha={a}')
            cells.append(fmt(m, c))
        blk, lab = ('fmnist_main', 'rfw_pgm|alpha=0.5|eta=0.3') if key == 'TGM full (lam=0.5)' else ('fmnist_ablation', f'{key}|alpha=0.5')
        m, c, f, n = cell(blk, lab)
        cells.append(fmt(m, c))
        tr = cell('ablation', f'{key}|alpha=0.5', 'trust_ratio')[0]
        cells.append(f'${tr:.2f}$' if tr < 100 else '$>100$')
        pre = '\\rowcolor{tgmhl}' if key == 'TGM full (lam=0.5)' else ''
        lines.append(f'{pre}{nm} & ' + ' & '.join(cells) + ' \\\\')
    write('tab_score', '\n'.join(lines))


def t_sens():
    CTX['table'] = 'sens'
    grid = [('Sharpness $\\gamma_c$', 'gamma_c', [1.0, 2.0, 4.0, 8.0]),
            ('Balance $\\lambda$', 'balance_lambda', [0.0, 0.25, 0.5, 0.75, 1.0]),
            ('Decay $\\beta$', 'decay_beta', [0.7, 0.8, 0.9, 0.95]),
            ('$\\hat h$ mis-scaling', 'hhat_scale', [0.25, 0.5, 1.0, 2.0, 4.0]),
            ('Warm-up $w$', 'warmup', [5, 10])]
    lines = []
    for nm, hp, vs in grid:
        vtxt = ' / '.join(f'{v:g}' for v in vs)
        b1 = ' / '.join(f"{cell('sensitivity', f'{hp}={v}|alpha=0.1')[0]:.1f}" for v in vs)
        b5 = ' / '.join(f"{cell('sensitivity', f'{hp}={v}|alpha=0.5')[0]:.1f}" for v in vs)
        f1 = sum(cell('sensitivity', f'{hp}={v}|alpha=0.1')[2] for v in vs)
        lines.append(f'{nm} & {vtxt} & {b1} & {b5} \\\\')
    write('tab_sens', '\n'.join(lines))


def figure_coords():
    """Coordinates for the three pgfplots figures."""
    out = []
    alphas = [0.05, 0.1, 0.5, 1.0]
    for d in ['fedavg', 'median', 'multikrum', 'alignins', 'fltrust', 'rfw', 'rfw_pgm']:
        pts = ''.join(f'({a},{cell("alpha_all", f"{d}|alpha={a}")[0]:.1f})' for a in alphas)
        out.append(f'alpha {d}: {pts}')
    for d in ['fedavg', 'median', 'alignins', 'fltrust', 'rfw_pgm']:
        pts = ' '.join(f'({r:.2f},{cell("participation", f"{d}|rho={r}")[0]:.1f})' for r in [0.2, 0.3, 0.4, 0.6])
        out.append(f'rho {d}: {pts}')
    for d in ['median', 'multikrum', 'bulyan', 'alignins', 'fltrust', 'rfw_pgm']:
        ms = [cell('eta_all', f'{d}|eta={e}') for e in [0.05, 0.1, 0.2, 0.3, 0.4]]
        mean = ''.join(f'({e},{m[0]:.1f})' for e, m in zip([0.05, 0.1, 0.2, 0.3, 0.4], ms))
        up = ''.join(f'({e},{min(100, m[0] + m[1]):.1f})' for e, m in zip([0.05, 0.1, 0.2, 0.3, 0.4], ms))
        lo = ''.join(f'({e},{max(0, m[0] - m[1]):.1f})' for e, m in zip([0.05, 0.1, 0.2, 0.3, 0.4], ms))
        out.append(f'eta {d}: mean {mean} up {up} lo {lo}')
    write('figure_coords', '\n'.join(out))


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['main', 'mainx', 'alpha', 'failure', 'participation', 'eta', 'attacks',
                             'adaptive', 'fmnist', 'fmnist2', 'unsw', 'quant', 'capacity',
                             'intermittent', 'warmup', 'ablation', 'score', 'sens', 'figs']
    fn = dict(main=t_main, mainx=t_mainx, alpha=t_alpha, failure=t_failure,
              participation=t_participation, eta=t_eta, attacks=t_attacks, adaptive=t_adaptive,
              fmnist=t_fmnist, fmnist2=t_fmnist2, unsw=t_unsw, quant=t_quant, capacity=t_capacity,
              intermittent=t_intermittent, warmup=t_warmup, ablation=t_ablation, score=t_score,
              sens=t_sens, figs=figure_coords)
    for w in which:
        try:
            fn[w]()
        except (KeyError, FileNotFoundError) as exc:
            print('SKIP', w, 'missing data:', exc)
    with open(os.path.join(OUT, 'changes_r1_to_r2.csv'), 'w') as f:
        f.write('table,R1,R2\n')
        for t, o, n in CHANGES:
            f.write(f'{t},"{o}","{n}"\n')
    print(len(CHANGES), 'R1 cells changed')
