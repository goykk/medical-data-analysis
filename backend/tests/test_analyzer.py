# medical-data-analysis/backend/tests/test_analyzer.py
"""文件解析、清洗、描述统计、正态性、组间比较与相关性的单元测试。

正确性验证策略：对每一项统计结果，用一条独立的库调用路径（scipy / statsmodels /
numpy）算出期望值，与被测函数返回值比对，而不是写死数字。
"""

import io

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

import analyzer
from analyzer import ParseError
from config import ALPHA

# ------------------------------------------------------------------
# 1. 文件解析
# ------------------------------------------------------------------

def test_parse_csv_utf8():
    content = "年龄,性别,空腹血糖\n55,1,5.6\n62,0,7.1\n".encode("utf-8")
    df, meta = analyzer.parse_table(content, "data.csv")
    assert list(df.columns) == ["年龄", "性别", "空腹血糖"]
    assert len(df) == 2
    assert meta["format"] == "csv"


def test_parse_csv_gbk_fallback():
    """GBK 编码文件：UTF-8 解码失败后应自动切换编码并成功读取。"""
    content = "年龄,性别,空腹血糖\n55,1,5.6\n62,0,7.1\n".encode("gbk")
    df, meta = analyzer.parse_table(content, "data.csv")
    assert list(df.columns) == ["年龄", "性别", "空腹血糖"]
    assert meta["encoding"].lower().startswith("gb")


def test_parse_tsv():
    content = "年龄\t性别\n55\t1\n62\t0\n".encode("utf-8")
    df, meta = analyzer.parse_table(content, "data.tsv")
    assert list(df.columns) == ["年龄", "性别"]
    assert meta["format"] == "tsv"


def test_parse_nan_literal_becomes_missing_and_numeric():
    """真实数据把 NaN 写成 'NaNN'：必须识别为缺失，并把整列转回数值 dtype。

    否则该列是 object，确认页可能仍判为 numeric，SimpleImputer(median) 会抛
    ValueError，整个 /api/ml/train 变成 500（Cirrhosis 数据集即如此）。
    """
    content = "年龄,胆红素\n55,1.2\n62,NaNN\n48,0.9\n".encode("utf-8")
    df, _ = analyzer.parse_table(content, "data.csv")
    assert pd.api.types.is_numeric_dtype(df["胆红素"])
    assert int(df["胆红素"].isna().sum()) == 1
    assert int(df["胆红素"].notna().sum()) == 2


def test_parse_mostly_numeric_column_with_rare_junk_coerced():
    """95% 以上可转数值的文本列视为数值列，零星脏值转缺失。"""
    rows = "\n".join(["id,x"] + [f"{i},{i}" for i in range(19)] + ["19,bad"])
    df, _ = analyzer.parse_table(rows.encode("utf-8"), "data.csv")
    assert pd.api.types.is_numeric_dtype(df["x"])
    assert int(df["x"].isna().sum()) == 1


def test_parse_text_category_kept_as_object():
    """非数值类别列不应被误转：'?' 变缺失，其余仍按文本类别保留。"""
    content = "id,性别\n1,男\n2,女\n3,?\n".encode("utf-8")
    df, _ = analyzer.parse_table(content, "data.csv")
    assert df["性别"].dtype == object
    assert int(df["性别"].isna().sum()) == 1
    assert set(df["性别"].dropna()) == {"男", "女"}


def test_parse_ragged_rows_skipped_with_note():
    """真实数据行字段数不齐：不能整份文件报错，应跳过坏行并在 meta 里如实告知。"""
    content = ("年龄,性别,血糖\n"
               "55,1,5.6\n"
               "62,0,7.1\n"
               "48,1,6.0,多出来的一列\n"
               "50,0,5.2\n").encode("utf-8")
    df, meta = analyzer.parse_table(content, "data.csv")
    assert len(df) == 3
    assert meta["skipped_rows"] == 1
    assert "已跳过" in meta["note"]


def test_parse_clean_csv_has_no_skipped_rows():
    content = "年龄,性别\n55,1\n62,0\n".encode("utf-8")
    df, meta = analyzer.parse_table(content, "data.csv")
    assert "skipped_rows" not in meta
    assert len(df) == 2


def test_parse_xlsx():
    buf = io.BytesIO()
    pd.DataFrame({"年龄": [55, 62], "空腹血糖": [5.6, 7.1]}).to_excel(buf, index=False)
    df, meta = analyzer.parse_table(buf.getvalue(), "data.xlsx")
    assert list(df.columns) == ["年龄", "空腹血糖"]
    assert meta["format"] == "xlsx"


def test_parse_json():
    content = b'[{"age": 55, "glucose": 5.6}, {"age": 62, "glucose": 7.1}]'
    df, meta = analyzer.parse_table(content, "data.json")
    assert len(df) == 2 and "age" in df.columns
    assert meta["format"] == "json"


def test_parse_headerless_numeric_file_detected():
    """无表头文件：首行是数据，不能被当成列名（否则列名变成 1.0000000e+000）。"""
    content = ("-1.7936000e-001\t-2.0700000e-001\t-2.0971000e-001\n"
               "-1.5000000e-001\t-2.1000000e-001\t-2.0000000e-001\n"
               "1.0000000e-001\t5.0000000e-002\t2.0000000e-002\n").encode("utf-8")
    df, meta = analyzer.parse_table(content, "plrx.txt")
    assert meta.get("headerless") is True
    assert "未检测到表头" in meta["note"]
    assert list(df.columns)[:3] == ["col_1", "col_2", "col_3"]
    assert len(df) == 3, "首行数据不能被吞掉"


def test_parse_normal_header_not_marked_headerless():
    content = "年龄,性别\n55,1\n62,0\n".encode("utf-8")
    df, meta = analyzer.parse_table(content, "data.csv")
    assert "headerless" not in meta
    assert list(df.columns) == ["年龄", "性别"]


def test_parse_headerless_symbolic_file_detected():
    """符号/序列型无表头文件（Promoter 风格）：列名是 '+' 与整段 DNA 序列。"""
    seq1 = "tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt"
    seq2 = "ttatgctgactgctagctagctagctagctagctagctagctagctagctagctagcta"
    seq3 = "gctagctagctagctagctagctagctagctagctagctagctagctagctagctagcta"
    content = f"+\tS10\t{seq1}\n-\tS11\t{seq2}\n+\tS12\t{seq3}\n".encode("utf-8")
    df, meta = analyzer.parse_table(content, "promoter.txt")
    assert meta.get("headerless") is True
    assert list(df.columns) == ["col_1", "col_2", "col_3"]
    assert len(df) == 3


def test_parse_sav_roundtrip():
    pyreadstat = pytest.importorskip("pyreadstat")
    src = pd.DataFrame({"年龄": [55, 62], "性别": [1, 0]})
    tmp = "tmp_roundtrip.sav"
    try:
        pyreadstat.write_sav(src, tmp)
        with open(tmp, "rb") as f:
            df, meta = analyzer.parse_table(f.read(), "data.sav")
    finally:
        import os
        if os.path.exists(tmp):
            os.remove(tmp)
    assert list(df.columns) == ["年龄", "性别"]
    assert meta["format"] == "sav"


def test_parse_unsupported_extension():
    with pytest.raises(ParseError) as exc:
        analyzer.parse_table(b"abc", "data.pdf")
    assert "不支持的文件格式" in str(exc.value)


def test_parse_broken_excel_raises():
    with pytest.raises(ParseError) as exc:
        analyzer.parse_table(b"this is definitely not an excel file", "broken.xlsx")
    assert "解析失败" in str(exc.value)


# ------------------------------------------------------------------
# 2. 数据清洗
# ------------------------------------------------------------------

def test_cleaning_report_detects_duplicates_and_missing(medical_df):
    import validators
    schema = validators.infer_variable_types(medical_df)
    rep = analyzer.build_cleaning_report(medical_df, schema)
    assert rep["n_duplicated_rows"] == 2
    assert any(m["field"] == "空腹血糖" and m["n_missing"] == 15 for m in rep["missing"])
    assert any(o["field"] == "空腹血糖" and o["n_above_reference"] >= 1 for o in rep["outliers"])
    assert any(f["field"] == "患者ID" for f in rep["excluded_fields"])


def test_apply_cleaning_drops_duplicates(medical_df):
    before = len(medical_df)
    df2, log = analyzer.apply_cleaning(medical_df, {"drop_duplicates": True})
    assert len(df2) == before - 2
    assert log["actions"][0]["action"] == "删除重复行"


def test_apply_cleaning_drops_out_of_reference_rows(medical_df):
    df2, _ = analyzer.apply_cleaning(medical_df, {"outlier_action": "drop_rows"})
    assert df2["空腹血糖"].max() <= 6.1


def test_apply_cleaning_noop(medical_df):
    df2, log = analyzer.apply_cleaning(medical_df, {})
    assert len(df2) == len(medical_df)
    assert log["actions"][0]["action"] == "未做修改"


def test_apply_cleaning_reports_drop_rate():
    """去重删掉大量行时必须显式给出比例与提示，不能静默。"""
    df = pd.DataFrame({"a": [1] * 7 + [2, 3, 4], "b": [1] * 7 + [2, 3, 4]})
    df2, log = analyzer.apply_cleaning(df, {"drop_duplicates": True})
    assert len(df2) == 4
    assert log["drop_rate"] > 0.5
    assert log["actions"][0]["rate"] > 0.5
    assert "占" in log["actions"][0]["detail"]
    assert "warning" in log


def test_unsupported_format_mentions_arff():
    """不支持的格式要明确说明 ARFF/影像/信号矩阵需先转表格，避免用户反复试。"""
    with pytest.raises(analyzer.ParseError) as exc:
        analyzer.parse_table(b"@relation x\n", "data.arff")
    assert "ARFF" in str(exc.value)


# ------------------------------------------------------------------
# 3. 描述性统计
# ------------------------------------------------------------------

def test_describe_numeric_matches_pandas(medical_df):
    s = medical_df["年龄"].dropna()
    res = analyzer.describe_numeric(medical_df, "年龄")
    assert res["n"] == len(s)
    assert res["mean"] == pytest.approx(float(s.mean()), abs=1e-3)
    assert res["sd"] == pytest.approx(float(s.std(ddof=1)), abs=1e-3)
    assert res["median"] == pytest.approx(float(np.median(s)), abs=1e-3)
    assert res["q1"] == pytest.approx(float(np.percentile(s, 25)), abs=1e-3)
    assert res["q3"] == pytest.approx(float(np.percentile(s, 75)), abs=1e-3)
    assert res["ci95"][0] < res["mean"] < res["ci95"][1]


def test_describe_categorical_percent_sum_to_one(medical_df):
    res = analyzer.describe_categorical(medical_df, "分组")
    assert sum(x["percent"] for x in res["levels"]) == pytest.approx(1.0, abs=1e-3)
    assert sum(x["count"] for x in res["levels"]) == res["n"]


def test_describe_numeric_all_missing_gives_note():
    df = pd.DataFrame({"x": [np.nan, np.nan]})
    res = analyzer.describe_numeric(df, "x")
    assert res["valid"] is False and "无有效数值" in res["note"]


# ------------------------------------------------------------------
# 4. 正态性检验
# ------------------------------------------------------------------

def test_normality_on_normal_sample():
    rng = np.random.default_rng(1)
    res = analyzer.normality_test(rng.normal(50, 8, 120))
    assert res["usable"] is True and res["normal"] is True
    assert res["reference_only"] is False


def test_normality_on_uniform_sample():
    rng = np.random.default_rng(2)
    res = analyzer.normality_test(rng.uniform(0, 100, 200))
    assert res["normal"] is False and res["p"] < ALPHA


def test_normality_large_sample_flagged_reference_only():
    rng = np.random.default_rng(3)
    res = analyzer.normality_test(rng.normal(0, 1, 600))
    assert res["n"] == 600 and res["reference_only"] is True
    assert "仅作参考" in res["note"]


def test_normality_too_few_samples():
    res = analyzer.normality_test([1.0, 2.0])
    assert res["usable"] is False


# ------------------------------------------------------------------
# 5. 单变量组间比较 —— 与独立库调用比对
# ------------------------------------------------------------------

def _two_group_df(a, b):
    return pd.DataFrame({"group": ["A"] * len(a) + ["B"] * len(b), "val": np.r_[a, b]})


def test_t_test_matches_scipy():
    rng = np.random.default_rng(10)
    a, b = rng.normal(10, 2, 60), rng.normal(12, 2, 60)
    res = analyzer.compare_groups(_two_group_df(a, b), "val", "group", "numeric")
    t, p = stats.ttest_ind(a, b, equal_var=True)
    assert "t 检验" in res["test"]["name"]
    assert res["test"]["statistic"] == pytest.approx(t, abs=1e-2)
    assert res["test"]["p"] == pytest.approx(p, abs=1e-3)
    assert res["estimate"]["value"] == pytest.approx(float(a.mean() - b.mean()), abs=1e-2)
    assert res["estimate"]["ci"][0] < res["estimate"]["value"] < res["estimate"]["ci"][1]
    assert res["effect"]["name"] == "Cohen's d"


def test_cohens_d_matches_manual_pooled_sd():
    """Cohen's d 必须等于 mean_diff / pooled_sd（独立手算核对）。

    CODE_REVIEW 曾把 `CompareMeans.std_meandiff_pooledvar` 误判为"合并方差"，
    实测它是均值差的**标准误** = pooled_sd × sqrt(1/n1+1/n2)，
    因此现有 `SE / sqrt(1/n1+1/n2)` 的换算正确。本用例锁定该结论，防止被误改。
    """
    rng = np.random.default_rng(13)
    a = rng.normal(10, 2, 40)
    b = rng.normal(12, 2, 55)
    res = analyzer.compare_groups(_two_group_df(a, b), "val", "group", "numeric")
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    expected_d = float(a.mean() - b.mean()) / sp
    assert res["effect"]["value"] == pytest.approx(expected_d, rel=0.02)


def test_welch_used_when_variance_unequal():
    rng = np.random.default_rng(11)
    a, b = rng.normal(10, 1, 50), rng.normal(10, 6, 50)
    res = analyzer.compare_groups(_two_group_df(a, b), "val", "group", "numeric")
    lev = stats.levene(a, b)
    assert lev.pvalue < ALPHA
    assert "Welch" in res["test"]["name"]
    lev_check = next(c for c in res["test"]["assumption_checks"] if "Levene" in c["name"])
    assert lev_check["passed"] is False


def test_mann_whitney_used_when_non_normal():
    rng = np.random.default_rng(12)
    a, b = rng.exponential(1.0, 45), rng.exponential(3.0, 45)
    res = analyzer.compare_groups(_two_group_df(a, b), "val", "group", "numeric")
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert "Mann-Whitney" in res["test"]["name"]
    assert res["test"]["statistic"] == pytest.approx(u, abs=1e-2)
    assert res["test"]["p"] == pytest.approx(p, abs=1e-3)


def test_anova_matches_statsmodels():
    rng = np.random.default_rng(13)
    a, b, c = rng.normal(10, 2, 40), rng.normal(12, 2, 40), rng.normal(14, 2, 40)
    df = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 40 + ["C"] * 40, "val": np.r_[a, b, c]})
    res = analyzer.compare_groups(df, "val", "group", "numeric")
    table = anova_lm(ols('Q("val") ~ C(Q("group"))', data=df).fit(), typ=2)
    assert "方差分析" in res["test"]["name"]
    assert res["test"]["statistic"] == pytest.approx(float(table.loc['C(Q("group"))', "F"]), abs=1e-2)
    assert res["test"]["p"] == pytest.approx(float(table.loc['C(Q("group"))', "PR(>F)"]), abs=1e-3)
    assert res["effect"]["name"] == "η²"


def test_alexander_govern_used_when_unequal_variance_multigroup():
    rng = np.random.default_rng(14)
    a, b, c = rng.normal(10, 1, 40), rng.normal(11, 2, 40), rng.normal(12, 6, 40)
    df = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 40 + ["C"] * 40, "val": np.r_[a, b, c]})
    res = analyzer.compare_groups(df, "val", "group", "numeric")
    assert "Alexander-Govern" in res["test"]["name"]


def test_kruskal_used_when_non_normal_multigroup():
    rng = np.random.default_rng(15)
    a, b, c = rng.exponential(1, 40), rng.exponential(2, 40), rng.exponential(3, 40)
    df = pd.DataFrame({"group": ["A"] * 40 + ["B"] * 40 + ["C"] * 40, "val": np.r_[a, b, c]})
    res = analyzer.compare_groups(df, "val", "group", "numeric")
    assert "Kruskal-Wallis" in res["test"]["name"]


def _binary_df(counts):
    rows = []
    for level, (n0, n1) in counts.items():
        rows += [{"group": level, "event": "否"}] * n0
        rows += [{"group": level, "event": "是"}] * n1
    return pd.DataFrame(rows)


def test_fisher_used_when_expected_frequency_lt_5():
    df = _binary_df({"A": (10, 1), "B": (4, 9)})
    res = analyzer.compare_groups(df, "event", "group", "categorical")
    assert "Fisher" in res["test"]["name"]
    assert "期望频数" in res["test"]["note"]
    assert res["estimate"]["label"] == "比值比 OR"
    assert res["estimate"]["ci"][0] < res["estimate"]["ci"][1]


def test_chi_square_matches_scipy():
    df = _binary_df({"A": (60, 40), "B": (40, 60)})
    res = analyzer.compare_groups(df, "event", "group", "categorical")
    table = pd.crosstab(df["group"], df["event"])
    chi2, p, _, _ = stats.chi2_contingency(table.to_numpy())
    assert "卡方检验" in res["test"]["name"]
    assert res["test"]["statistic"] == pytest.approx(chi2, abs=1e-2)
    assert res["test"]["p"] == pytest.approx(p, abs=1e-3)


def test_cramers_v_matches_scipy():
    df = _binary_df({"A": (60, 40), "B": (40, 60), "C": (30, 70)})
    res = analyzer.compare_groups(df, "event", "group", "categorical")
    expected = stats.contingency.association(pd.crosstab(df["group"], df["event"]).to_numpy(),
                                             method="cramer")
    assert res["effect"]["value"] == pytest.approx(float(expected), abs=1e-3)


def test_compare_groups_insufficient_data():
    df = pd.DataFrame({"group": ["A", "B"], "val": [1.0, 2.0]})
    res = analyzer.compare_groups(df, "val", "group", "numeric")
    assert res["valid"] is False


# ------------------------------------------------------------------
# 6. 相关性与 FDR
# ------------------------------------------------------------------

def test_correlation_matches_scipy_pearsonr():
    rng = np.random.default_rng(20)
    x = rng.normal(0, 1, 120)
    y = x * 0.8 + rng.normal(0, 0.4, 120)
    df = pd.DataFrame({"x": x, "y": y})
    res = analyzer.correlation_matrix(df, ["x", "y"])
    r, p = stats.pearsonr(x, y)
    assert res["r"][0][1] == pytest.approx(float(r), abs=1e-3)
    assert res["p"][0][1] == pytest.approx(float(p), abs=1e-4)
    assert res["method"][0][1] == "pearson"


def test_correlation_uses_spearman_for_non_normal():
    rng = np.random.default_rng(21)
    x = rng.exponential(2, 150)
    y = x + rng.exponential(2, 150)
    res = analyzer.correlation_matrix(pd.DataFrame({"x": x, "y": y}), ["x", "y"])
    assert res["method"][0][1] == "spearman"


def test_correlation_constant_field_does_not_crash():
    """常量列（或成对删除后只剩单一取值）产生 NaN p，相关矩阵必须正常返回。

    回归：旧实现把 None 混入 multipletests，'<' not supported between
    instances of 'NoneType' and 'float'，接口 500。
    """
    rng = np.random.default_rng(31)
    df = pd.DataFrame({"x": rng.normal(size=50), "const": 3.0,
                       "sparse": [1.0] * 30 + [np.nan] * 20})
    res = analyzer.correlation_matrix(df, ["x", "const", "sparse"])
    assert res["p"][0][1] is None
    assert res["p"][0][2] is None
    assert res["p_fdr"][0][1] is None
    assert res["p_fdr"][0][2] is None
    # 不应把无效 pair 的 p 当作 0 或 1 参与校正
    assert res["r"][0][1] is None


def test_fdr_matches_statsmodels():
    pvals = [0.001, 0.01, 0.04, 0.3, 0.6]
    res = analyzer.fdr_correct(pvals)
    _, corrected, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    assert res["p_fdr"] == pytest.approx([round(v, 6) for v in corrected], abs=1e-6)


# ------------------------------------------------------------------
# 7. 绘图数据
# ------------------------------------------------------------------

def test_histogram_and_boxplot_shapes(medical_df):
    hist = analyzer.histogram_data(medical_df, "年龄", bins=10)
    assert len(hist["counts"]) == 10 and len(hist["edges"]) == 11
    assert sum(hist["counts"]) == hist["n"]
    box = analyzer.boxplot_data(medical_df, "年龄")
    assert box["q1"] <= box["median"] <= box["q3"]
    assert isinstance(box["outliers"], list)


def test_qqplot_matches_probplot(medical_df):
    s = medical_df["年龄"].dropna()
    res = analyzer.qqplot_data(medical_df, "年龄")
    (osm, osr), _ = stats.probplot(s.to_numpy(), dist="norm")
    assert res["sample"][0] == pytest.approx(float(osr[0]), abs=1e-6)
    assert res["theoretical"][0] == pytest.approx(float(osm[0]), abs=1e-6)
