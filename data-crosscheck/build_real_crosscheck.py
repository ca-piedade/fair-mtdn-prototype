"""
build_real_crosscheck.py
------------------------
Reconstrucao do consumo teorico F&B a partir de (a) vendas POS do host e
(b) fichas tecnicas, e cruzamento com o consumo real registado no ERP.

Janela: 18-05-2026 a 28-06-2026 (a mesma do hold-out temporal da tese).

Entradas esperadas (nao versionadas -- ver .gitignore):
  30. Vendas por Artigo_1805A2806.xlsx          (host, vendas por artigo POS)
  98b. Fichas Tecnicas Detalhado 08072026.xlsx  (1 folha por ficha tecnica)
  Listage_Consumos_por_Artigo_ERP.xlsx          (ERP, consumo por ref de stock)
  Consumos_Relatorio_de_Diferencas_de_Consumos_Host_Vs_PHC_1805a2806.xlsx

Saidas (agregadas, versionaveis):
  fichas_tecnicas_flat.csv        explosao ficha tecnica -> ingrediente
  cruzamento_teorico_vs_erp.csv   consumo teorico vs real por ref ERP
  discrepancy_log_summary.csv     resumo do log de 3.705 discrepancias

Nota metodologica: o consumo teorico por unidade vendida e' qtd_ingrediente/pax,
porque a ficha tecnica exprime quantidades para um numero de pax declarado.
"""
import openpyxl, csv, json, re, argparse, os
from collections import defaultdict

HDR = re.compile(r'^\s*(\d+)\s+-\s+(.*)$')


def parse_fichas(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    recipes = {}
    for ws in wb.worksheets:
        cur, mode = None, None
        for r in ws.iter_rows(values_only=True):
            if not r or all(c in (None, '') for c in r):
                continue
            c1 = str(r[1]).strip() if len(r) > 1 and r[1] not in (None, '') else ''
            if c1 == 'Recipe':
                mode = 'recipe'; continue
            if c1 == 'Ficha técnica':
                mode = None; continue
            m = HDR.match(c1) if c1 else None
            if m and (len(r) < 8 or r[7] in (None, '')):
                cur = m.group(1)
                recipes.setdefault(cur, {'desc': m.group(2).strip(), 'ing': []})
                mode = None; continue
            if mode == 'recipe' and c1.replace(' ', '').isdigit() and len(c1.strip()) == 11:
                if cur is None:
                    continue
                recipes[cur]['ing'].append(dict(
                    ref=c1.strip(), desc=str(r[5]).strip() if r[5] else '',
                    qtd=r[7], un=r[8], pax=r[9] or 1,
                    custo_pax=r[12] if len(r) > 12 else None))
    wb.close()
    return recipes


def parse_vendas(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True)); wb.close()
    agg = defaultdict(lambda: [0.0, 0.0, 0.0])   # qtd, valor, custo
    for r in rows[11:]:
        c2, c3, c6 = r[1], r[2], r[5]
        if c2 in (None, ''):
            continue
        s = str(c2)
        if s.startswith('Sector:') or s == 'Total':
            continue
        if c3 not in (None, '') and c6 not in (None, ''):   # linha de artigo
            v = agg[s]
            v[0] += r[6] or 0; v[1] += r[7] or 0; v[2] += r[12] or 0
    return agg


def parse_erp(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True)); wb.close()
    erp = {}
    for r in rows[5:]:
        if r[1] in (None, ''):
            continue
        erp[str(r[1]).strip()] = dict(
            desc=str(r[2]).strip(), fam=str(r[3]).strip(), un=str(r[4]).strip(),
            val=r[5] or 0, qt=r[6] or 0, unalt=str(r[9]).strip(),
            fator=r[10] or 1, qtalt=r[11] or 0)
    return erp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vendas', required=True)
    ap.add_argument('--fichas', required=True)
    ap.add_argument('--erp', required=True)
    ap.add_argument('--outdir', default='.')
    a = ap.parse_args()

    recipes = parse_fichas(a.fichas)
    vendas = parse_vendas(a.vendas)
    erp = parse_erp(a.erp)

    with open(os.path.join(a.outdir, 'fichas_tecnicas_flat.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pos_code', 'pos_desc', 'erp_ref', 'erp_desc', 'qtd', 'unidade', 'pax', 'custo_pax'])
        for k, v in recipes.items():
            for g in v['ing']:
                w.writerow([k, v['desc'], g['ref'], g['desc'], g['qtd'], g['un'], g['pax'], g['custo_pax']])

    theo_q, theo_v, theo_u = defaultdict(float), defaultdict(float), {}
    for code, (q, _val, _cu) in vendas.items():
        rec = recipes.get(code)
        if not rec or not rec['ing']:
            continue
        for g in rec['ing']:
            pax = g['pax'] or 1
            theo_q[g['ref']] += q * (g['qtd'] or 0) / pax
            theo_v[g['ref']] += q * (g['custo_pax'] or 0)
            theo_u[g['ref']] = g['un']

    with open(os.path.join(a.outdir, 'cruzamento_teorico_vs_erp.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ref_erp', 'designacao', 'familia', 'unidade_ficha', 'qtd_teorica',
                    'qtd_erp', 'match_unidade', 'valor_teorico', 'valor_erp', 'variancia_valor'])
        for ref in sorted(set(theo_q) & set(erp)):
            e = erp[ref]; u = (theo_u[ref] or '').strip()
            if u == e['un']:
                qe, how = e['qt'], 'base'
            elif u == e['unalt']:
                qe, how = e['qtalt'], 'alt'
            else:
                qe, how = e['qt'], 'incompativel'
            w.writerow([ref, e['desc'], e['fam'], u, round(theo_q[ref], 4), round(qe, 4),
                        how, round(theo_v[ref], 2), round(e['val'], 2),
                        round(e['val'] - theo_v[ref], 2)])

    cov = {c for c in vendas if recipes.get(c, {}).get('ing')}
    tq = sum(v[0] for v in vendas.values()); tv = sum(v[1] for v in vendas.values())
    cq = sum(vendas[c][0] for c in cov); cv = sum(vendas[c][1] for c in cov)
    print(f'codigos POS vendidos ............ {len(vendas)}')
    print(f'  com ficha tecnica utilizavel .. {len(cov)} ({100*len(cov)/len(vendas):.0f}%)')
    print(f'  cobertura em quantidade ....... {100*cq/tq:.1f}%')
    print(f'  cobertura em valor de vendas .. {100*cv/tv:.1f}%')
    print(f'valor teorico reconstruido ...... {sum(theo_v.values()):.2f}')
    print(f'custo teorico reportado pelo host {sum(v[2] for v in vendas.values()):.2f}')


if __name__ == '__main__':
    main()
