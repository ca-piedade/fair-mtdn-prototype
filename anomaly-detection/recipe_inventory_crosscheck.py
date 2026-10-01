"""
FAIR artefact - recipe-versus-consumption reconciliation and cross-source
correlation check (Section 4.5.3 of the project). Reconstructed on 26 Sep 2026
from the raw exports so that the Section 4.5.3 figures can be reproduced; the
original exploratory run (24 Aug 2026) was not saved as a script.

Inputs (operational data of the host organisation - not included in the repository)
------------------------------------------------------------------------------------
--sales      POS report "30. Vendas por Artigo" (18 May - 28 Jun 2026), units sold per
             POS dish code.
--recipes    fichas_tecnicas_flat.csv: technical sheets exploded to ingredient level
             (pos_code, erp_ref, qtd, unidade, pax), produced by
             data-crosscheck/build_real_crosscheck.py.
--consumption real_dataset_v2.csv: real ERP consumption lines for the same window
             (client, internal/complimentary and the lines deleted at integration),
             with each article's historical quantity standard deviation (qty_std).
--inventory  folder with the 48 physical-count "faltas e excessos" exports.

Steps
-----
1. Theoretical consumption per ingredient = sum over sold dishes of
   units sold x ingredient quantity / pax.
2. Recipe reconciliation: ratio = actual ERP consumption / theoretical consumption,
   for ingredients with a positive theoretical quantity that appear in the
   consumption record. Reported: number of comparable ingredients, median ratio,
   share within +/-20%.
3. Cross-source check at ingredient x count granularity (one row per ingredient per
   physical-count file): point-biserial correlation between the real inventory
   discrepancy label (physical != book) and (a) the ingredient's recipe deviation
   |ratio - 1|, (b) the ingredient's consumption volatility (historical standard
   deviation of consumed quantity). Alternative specifications are printed as a
   sensitivity check.

Run
---
    pip install --break-system-packages pandas numpy scipy openpyxl
    python3 recipe_inventory_crosscheck.py --sales <xlsx> --recipes <csv> \
        --consumption <csv> --inventory <folder>
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


def load_sales(path):
    raw = pd.read_excel(path, header=None).iloc[11:]
    rows = raw[raw[1].notna() & raw[2].notna() & raw[5].notna()]
    rows = rows[~rows[1].astype(str).str.startswith("Sector")]
    rows = rows.assign(code=rows[1].astype(str).str.strip(),
                       qty=pd.to_numeric(rows[6], errors="coerce").fillna(0))
    return rows.groupby("code")["qty"].sum()


def theoretical(sales, recipes_path):
    ft = pd.read_csv(recipes_path, dtype=str)
    ft["qtd"] = pd.to_numeric(ft["qtd"], errors="coerce").fillna(0)
    ft["pax"] = pd.to_numeric(ft["pax"], errors="coerce").fillna(1).replace(0, 1)
    linked = sorted(set(sales.index) & set(ft["pos_code"]))
    exploded = ft[ft["pos_code"].isin(linked)].merge(sales.rename("sold"), left_on="pos_code", right_index=True)
    exploded["theo"] = exploded["sold"] * exploded["qtd"] / exploded["pax"]
    return linked, exploded.groupby("erp_ref")["theo"].sum()


def load_consumption(path):
    df = pd.read_csv(path)
    df = df[df["Ref"].notna()].copy()
    df["ref"] = df["Ref"].astype("int64").astype(str).str.zfill(11)
    return df


def load_inventory(folder):
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, "*.xlsx"))):
        raw = pd.read_excel(path, header=None)
        hdr = raw.index[raw.apply(lambda r: r.astype(str).str.contains("Referencia").any(), axis=1)][0]
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c) for c in raw.iloc[hdr]]
        df = df[df["Referencia"].notna()]
        df["file"] = os.path.basename(path)
        frames.append(df)
    inv = pd.concat(frames, ignore_index=True)
    for col in ["Inv fisico", "Inv contabilistico"]:
        inv[col] = pd.to_numeric(inv[col], errors="coerce").fillna(0.0)
    inv["Referencia"] = inv["Referencia"].astype(str)
    inv["label"] = (inv["Inv fisico"] != inv["Inv contabilistico"]).astype(int)
    return inv


def pb(inv, series):
    joined = inv[["Referencia", "label"]].merge(series.rename("x"), left_on="Referencia", right_index=True).dropna()
    r, p = pointbiserialr(joined["label"], joined["x"])
    return len(joined), r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", required=True)
    ap.add_argument("--recipes", required=True)
    ap.add_argument("--consumption", required=True)
    ap.add_argument("--inventory", required=True)
    a = ap.parse_args()

    sales = load_sales(a.sales)
    linked, theo = theoretical(sales, a.recipes)
    theo = theo[theo > 0]
    print(f"POS dish codes sold: {len(sales)} | linked to a technical sheet: {len(linked)} | "
          f"theoretical ingredients: {len(theo)}")

    cons = load_consumption(a.consumption)
    actual = cons.groupby("ref")["Quant"].sum()
    rec = pd.concat([theo.rename("theo"), actual.rename("actual")], axis=1, join="inner")
    rec["ratio"] = rec["actual"] / rec["theo"]
    within = ((rec["ratio"] - 1).abs() <= 0.20).mean()
    print(f"Comparable ingredients: {len(rec)} | median actual/theoretical: {rec['ratio'].median():.3f} | "
          f"within +/-20%: {within:.1%}")

    inv = load_inventory(a.inventory)
    recipe_dev = (rec["ratio"] - 1).abs()
    volatility = cons.groupby("ref")["qty_std"].first()
    print("\nCross-source check (ingredient x physical count; label = physical != book):")
    for name, s in [("recipe deviation |ratio-1|", recipe_dev), ("consumption volatility (historical SD)", volatility)]:
        n, r, p = pb(inv, s)
        print(f"  {name:42s} n={n:5d}  r={r:+.3f}  p={p:.3f}")

    print("\nSensitivity (alternative specifications, not the reported ones):")
    g = cons.groupby("ref")["Quant"]
    alt = [("recipe ratio (signed)", rec["ratio"]),
           ("recipe log-ratio", np.log(rec["ratio"].where(rec["ratio"] > 0))),
           ("consumption SD of window quantities", g.std()),
           ("consumption coefficient of variation", g.std() / g.mean())]
    for name, s in alt:
        n, r, p = pb(inv, s)
        print(f"  {name:42s} n={n:5d}  r={r:+.3f}  p={p:.3g}")


if __name__ == "__main__":
    main()
