# medical-data-analysis/sample_data/generate_sample_data.py
"""生成内置示例数据（全部为模拟脱敏数据，不含任何真实患者信息）。

产出：
  1. medical_sample.csv   —— 模拟医疗数据集，GBK 编码（用于验证编码自动检测）
  2. medical_sample.xlsx  —— 同一数据集的 Excel 版本
  3. wdbc.csv / wdbc.xlsx —— WDBC 乳腺癌数据集导出（统计与建模全流程基准测试数据）

运行：python sample_data/generate_sample_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def make_medical_sample(n: int = 300, seed: int = SEED) -> pd.DataFrame:
    """模拟一项代谢综合征与不良结局关系的横断面研究（含缺失值、异常值、ID 列、常量列）。"""
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 10, n).round(0)
    gender = rng.integers(0, 2, n)
    bmi = rng.normal(24, 3, n).round(1)
    sbp = rng.normal(130, 15, n).round(0)
    dbp = (sbp * 0.62 + rng.normal(0, 6, n)).round(0)
    glucose = rng.normal(6.0, 1.5, n).round(2)
    tc = rng.normal(4.8, 1.0, n).round(2)
    smoking = rng.integers(0, 2, n)
    group = rng.choice(["对照组", "试验组"], n)

    # 结局由血糖、年龄、吸烟、分组共同生成（含真实效应，便于验证建模流程）
    logit = (-0.4 + 0.9 * (glucose - 6.0) + 0.03 * (age - 55)
             + 0.8 * smoking + 0.5 * (group == "试验组"))
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    df = pd.DataFrame({
        "患者ID": [f"P{i:04d}" for i in range(n)],
        "年龄": age,
        "性别": gender,
        "体质指数": bmi,
        "收缩压": sbp,
        "舒张压": dbp,
        "空腹血糖": glucose,
        "总胆固醇": tc,
        "吸烟": smoking,
        "分组": group,
        "结局": outcome,
        "研究中心": "A",
    })

    # 人为引入数据质量问题，用于验证清洗流程
    idx = rng.choice(n, size=15, replace=False)
    df.loc[idx, "空腹血糖"] = np.nan                       # 缺失值
    df.loc[rng.choice(n, size=8, replace=False), "总胆固醇"] = np.nan
    df.loc[0, "空腹血糖"] = 18.6                            # 超出参考区间的异常值
    df.loc[1, "收缩压"] = 215.0                             # 超出参考区间的异常值
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # 2 条重复行
    return df


def make_wdbc() -> pd.DataFrame:
    """导出 WDBC（Wisconsin Diagnostic Breast Cancer）数据集。

    sklearn.datasets.load_breast_cancer 即为 WDBC 的 569 例、30 特征版本，
    字段按 WDBC 原始命名规范重排为 radius_mean ... fractal_dimension_worst。
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    prefix_map = {"mean": "mean", "error": "se", "worst": "worst"}
    columns = []
    for name in data.feature_names:
        head, _, tail = name.partition(" ")
        suffix = prefix_map.get(head, head)
        base = tail.replace(" ", "_")
        columns.append(f"{base}_{suffix}")

    df = pd.DataFrame(data.data, columns=columns)
    rng = np.random.default_rng(SEED)
    df.insert(0, "id", [f"{i + 842302:07d}" for i in range(len(df))])
    df.insert(1, "diagnosis", np.where(data.target == 0, "M", "B"))  # 0=malignant
    return df


def main():
    medical = make_medical_sample()
    csv_path = os.path.join(OUT_DIR, "medical_sample.csv")
    xlsx_path = os.path.join(OUT_DIR, "medical_sample.xlsx")
    medical.to_csv(csv_path, index=False, encoding="gbk")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        medical.to_excel(w, index=False, sheet_name="模拟医疗数据")

    wdbc = make_wdbc()
    wdbc_csv = os.path.join(OUT_DIR, "wdbc.csv")
    wdbc_xlsx = os.path.join(OUT_DIR, "wdbc.xlsx")
    wdbc.to_csv(wdbc_csv, index=False, encoding="utf-8")
    with pd.ExcelWriter(wdbc_xlsx, engine="openpyxl") as w:
        wdbc.to_excel(w, index=False, sheet_name="WDBC")

    print("已生成：")
    for p in (csv_path, xlsx_path, wdbc_csv, wdbc_xlsx):
        print(f"  {os.path.basename(p):<24} {os.path.getsize(p) / 1024:8.1f} KB")
    print(f"\n模拟医疗数据：{medical.shape[0]} 行 × {medical.shape[1]} 列（GBK 编码 CSV + XLSX）")
    print(f"WDBC 乳腺癌数据：{wdbc.shape[0]} 行 × {wdbc.shape[1]} 列"
          f"（恶性 {(wdbc['diagnosis'] == 'M').sum()} 例 / 良性 {(wdbc['diagnosis'] == 'B').sum()} 例）")


if __name__ == "__main__":
    sys.exit(main())
