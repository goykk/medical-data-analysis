# medical-data-analysis/backend/tests/test_cleaning_enhancements.py
"""清洗管线增强：单元格 trim、缺失标记扩展（?）、高缺失列警示。

雷点来自 UCI 真实临床数据集：
- class 列混入 "ckd\t" 导致凭空多出第三类
- "?" 被当成合法字符串值而非缺失
- 近全缺失列无任何提示

验收重点：trim 必须在类型推断之前；高缺失只警示、不自动排除。
"""

import io

import numpy as np
import pandas as pd
import pytest

import analyzer
import validators
from config import HIGH_MISSING_THRESHOLD

from conftest import make_medical_df


def parse(csv_text: str, name="t.csv"):
    """经完整解析链路（读取 → trim → 缺失标记识别）得到 DataFrame。"""
    df, meta = analyzer.parse_table(csv_text.encode("utf-8"), name)
    return df, meta


def schema_of(df):
    return {v["name"]: v for v in validators.infer_variable_types(df)}


# ------------------------------------------------------------------
# 1. 脏类别：带制表符的值不得生成第三类
# ------------------------------------------------------------------
def test_trailing_tab_does_not_create_extra_category():
    csv = 'class,age,bp\nckd,35,80\n"ckd\t",36,82\nnotckd,37,84\n'
    df, _ = parse(csv)

    sch = schema_of(df)
    assert sch["class"]["n_unique"] == 2, sch["class"]

    values = sorted(set(df["class"].dropna().astype(str)))
    assert values == ["ckd", "notckd"], values


# ------------------------------------------------------------------
# 2. "?" 计为缺失，其余值正常转数值
# ------------------------------------------------------------------
def test_question_mark_is_treated_as_missing():
    csv = "age,bp\n35,80\n?,82\n42,?\n"
    df, _ = parse(csv)

    sch = schema_of(df)
    assert sch["age"]["n_missing"] == 1
    assert sch["age"]["inferred_type"] == "numeric", sch["age"]
    # missing_rate 后端保留 4 位小数
    assert abs(sch["age"]["missing_rate"] - round(1 / 3, 4)) < 1e-9

    assert sch["bp"]["n_missing"] == 1
    assert sch["bp"]["inferred_type"] == "numeric", sch["bp"]

    # 非缺失值仍可被正常转为数值
    assert sorted(df["age"].dropna().astype(float).tolist()) == [35.0, 42.0]


# ------------------------------------------------------------------
# 3. trim 时机：带空白的数值列应被推断为数值型
# ------------------------------------------------------------------
def test_whitespace_padded_number_is_numeric():
    # 用非整数值：少量整数值会被 _is_numeric_code 判为数字编码分类（那是既有行为，不改）
    rows = ['" 35.5 ",A', '"42.3\t",B', '" 35.5 ",A', '"51.1 ",C', '"42.3\t",B',
            '" 35.5 ",A', '"51.1 ",C', '"42.3\t",B', '" 35.5 ",A', '"51.1 ",C']
    csv = "age,group\n" + "\n".join(rows) + "\n"
    df, _ = parse(csv)

    # trim 后不得残留空白
    assert sorted(set(df["age"].dropna().astype(str))) == ["35.5", "42.3", "51.1"]

    sch = schema_of(df)
    assert sch["age"]["inferred_type"] == "numeric", sch["age"]


# ------------------------------------------------------------------
# 4. 仅含空白的单元格按缺失计数
# ------------------------------------------------------------------
def test_whitespace_only_cell_counts_as_missing():
    csv = "age,name\n35,   \n42,bob\n"
    df, _ = parse(csv)

    sch = schema_of(df)
    assert sch["name"]["n_missing"] == 1, sch["name"]
    assert pd.isna(df.loc[0, "name"])
    assert df.loc[1, "name"] == "bob"


# ------------------------------------------------------------------
# 5. 高缺失边界：>80% 警示，正常列不标记
# ------------------------------------------------------------------
def test_high_missing_flag_threshold():
    n = 20
    cols = {
        # 90% 缺失（18/20），保留 2 个不同值避免被判为常量列
        "mostly_missing": [np.nan] * 18 + ["p", "q"],
        # 50% 缺失
        "half_missing": [np.nan] * 10 + [f"v{i}" for i in range(10)],
        # 0% 缺失
        "complete": [f"c{i}" for i in range(n)],
    }
    df = pd.DataFrame(cols)
    sch = schema_of(df)

    assert sch["mostly_missing"]["high_missing"] is True
    assert sch["mostly_missing"]["excluded"] is False, "高缺失只警示，不自动排除"
    assert abs(sch["mostly_missing"]["missing_rate"] - 0.9) < 1e-6

    assert sch["half_missing"]["high_missing"] is False
    assert sch["complete"]["high_missing"] is False

    # 阈值常量本身符合预期
    assert HIGH_MISSING_THRESHOLD == 0.8


def test_high_missing_uses_strictly_greater_than():
    """恰好等于阈值不应触发警示（严格大于）。"""
    n = 10
    df = pd.DataFrame({
        "at_threshold": [np.nan] * 8 + ["a", "b"],   # 80% = 阈值
        "just_over": [np.nan] * 9 + ["a"],           # 90%
    })
    sch = schema_of(df)
    assert sch["at_threshold"]["high_missing"] is False
    assert sch["just_over"]["high_missing"] is True


# ------------------------------------------------------------------
# 6. 回归保护：干净数据零副作用
# ------------------------------------------------------------------
def test_clean_data_is_untouched():
    src = make_medical_df(n=120, seed=3, with_issues=True)
    buf = io.StringIO()
    src.to_csv(buf, index=False, encoding="utf-8")
    df, meta = parse(buf.getvalue())

    # 行数/列数不变
    assert len(df) == len(src)
    assert list(df.columns) == list(src.columns)

    # 数值列取值与缺失位置完全一致（trim/NA 扩展对干净数据无副作用）
    for col in ["年龄", "体质指数", "收缩压", "空腹血糖"]:
        a = src[col].astype(float)
        b = df[col].astype(float)
        assert a.isna().sum() == b.isna().sum(), col
        assert np.allclose(a.dropna().values, b.dropna().values), col

    # 分类列取值不变
    assert sorted(set(df["分组"].dropna().astype(str))) == sorted(set(src["分组"].astype(str)))

    sch = schema_of(df)
    assert sch["年龄"]["inferred_type"] == "numeric"
    assert sch["分组"]["inferred_type"] == "categorical"
    # 干净数据不应出现高缺失列
    assert not any(v["high_missing"] for v in sch.values())


# ------------------------------------------------------------------
# 7. 接口层：确认页响应携带高缺失标记
# ------------------------------------------------------------------
def test_confirm_response_exposes_high_missing():
    from fastapi.testclient import TestClient

    import app as app_module

    n = 20
    df = pd.DataFrame({
        "年龄": [float(i) for i in range(n)],
        "备注": [np.nan] * 18 + ["a", "b"],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")

    client = TestClient(app_module.app)
    r = client.post("/api/dataset/upload",
                    files={"file": ("hm.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    did = r.json()["dataset_id"]

    schema = client.get(f"/api/dataset/{did}").json()["variables"]
    by_name = {v["name"]: v for v in schema}
    assert by_name["备注"]["high_missing"] is True
    assert by_name["备注"]["excluded"] is False, "高缺失不得自动排除"
    assert by_name["备注"]["n_missing"] == 18
    assert by_name["年龄"]["high_missing"] is False
