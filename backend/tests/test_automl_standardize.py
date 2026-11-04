# medical-data-analysis/backend/tests/test_automl_standardize.py
"""模块零（列名标准化）测试：六类列名 + 需求列明的全部边界用例。

每条边界用例均写明期望行为断言，与《01_模块零_列名标准化_接口设计.md》对应。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import automl.standardize as std
from automl.errors import DataShapeError
from automl.standardize import (Lexicon, LexiconEntry, clean_column_name,
                                load_global_lexicon, standardize_dataframe)


def _rng_df(cols: dict, n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({k: rng.normal(size=n) for k in cols})


def _by_raw(result, raw):
    for c in result.columns:
        if c.raw_name == raw:
            return c
    raise AssertionError(f"映射结果中找不到列 {raw}")


# ------------------------------------------------------------------
# 列名清洗
# ------------------------------------------------------------------
def test_clean_full_width_and_case():
    c = clean_column_name("ＷＢＣ ")
    assert c.normalized == "wbc"          # 全角转半角 + 去空格 + 小写


def test_clean_unit_paren_is_stripped():
    c = clean_column_name("血糖(mmol/L)")
    assert c.unit_stripped == "mmol/l"
    assert c.base == "血糖"
    assert c.qualifier is None


def test_clean_non_unit_paren_kept_as_qualifier():
    # 期望："餐后2小时"保留为限定词，不得与"血糖"合并
    c = clean_column_name("血糖(餐后2小时)")
    assert c.qualifier == "餐后2小时"
    assert c.base == "血糖"
    assert c.normalized == "血糖(餐后2小时)"   # 全名保留限定词


def test_clean_numeric_paren_is_ordinal_not_qualifier():
    c = clean_column_name("白细胞(2)")
    assert c.ordinal_suffix == "2"
    assert c.qualifier is None
    assert c.base == "白细胞"


def test_clean_ordinal_prefix():
    c = clean_column_name("第3次肌钙蛋白")
    assert c.ordinal_suffix == "第3次"
    assert c.base == "肌钙蛋白"


def test_clean_ordinal_underscore_suffix():
    c = clean_column_name("白细胞_2")
    assert c.ordinal_suffix == "2"
    assert c.base == "白细胞"


def test_clean_blank_name_raises():
    with pytest.raises(DataShapeError):
        clean_column_name("   ")


# ------------------------------------------------------------------
# 六类列名（集成）
# ------------------------------------------------------------------
def test_six_column_classes():
    df = _rng_df(["白细胞", "WBC", "HGB血红蛋白", "血糖(mmol/L)",
                  "白细胞_2", "第3次肌钙蛋白", "JYZD001"])
    res = standardize_dataframe(df)

    # 中文简称
    assert _by_raw(res, "白细胞").standard_name == "白细胞计数"
    assert _by_raw(res, "白细胞").status == "auto"
    # 英文缩写：L1 原名精确
    assert _by_raw(res, "WBC").standard_name == "白细胞计数"
    assert _by_raw(res, "WBC").level == "L1"
    # 中英混合：清洗后 L2 命中
    assert _by_raw(res, "HGB血红蛋白").standard_name == "血红蛋白"
    assert _by_raw(res, "HGB血红蛋白").level == "L2"
    # 带单位
    g = _by_raw(res, "血糖(mmol/L)")
    assert g.standard_name == "血糖"
    # 带序号：命中对应标准名且保留序号
    assert _by_raw(res, "白细胞_2").standard_name == "白细胞计数"
    assert _by_raw(res, "第3次肌钙蛋白").standard_name == "肌钙蛋白I"
    # 内部编码：L5 未知列，保留为普通特征
    jz = _by_raw(res, "JYZD001")
    assert jz.level == "L5" and jz.status == "unknown" and jz.standard_name is None
    assert "JYZD001" in res.unknown_columns


# ------------------------------------------------------------------
# 边界用例（需求逐条）
# ------------------------------------------------------------------
def test_boundary_qualifier_is_independent_standard():
    """非单位括号保留：血糖(餐后2小时) ≠ 血糖，且不挂空腹参考区间。"""
    df = _rng_df(["血糖", "血糖(餐后2小时)", "餐后2小时血糖"])
    res = standardize_dataframe(df)
    fasting = _by_raw(res, "血糖")
    post = _by_raw(res, "血糖(餐后2小时)")
    alias = _by_raw(res, "餐后2小时血糖")
    assert fasting.standard_name == "血糖"
    assert post.standard_name == "血糖(餐后2小时)"       # 独立标准名
    assert alias.standard_name == "血糖(餐后2小时)"
    assert post.standard_name != fasting.standard_name
    # 餐后条目不得挂空腹血糖的参考区间
    fasting_entry = load_global_lexicon().by_standard["血糖"]
    post_entry = load_global_lexicon().by_standard["血糖(餐后2小时)".lower()]
    assert fasting_entry.reference_key == "glucose"
    assert post_entry.reference_key is None


def test_boundary_pinyin_initials_ambiguity_goes_pending():
    """拼音首字母歧义：多候选一律降级待确认，不自动映射。"""
    fake = Lexicon([
        LexiconEntry(standard_name="性别", pinyin_initials="xb"),
        LexiconEntry(standard_name="细胞计数", pinyin_initials="xb"),
    ])
    m = std.match_column(clean_column_name("XB"), fake, {})
    assert m.status == "pending"
    assert m.standard_name is None
    assert "多个候选" in m.confirm_reason
    assert len(m.candidates) == 2
    # 唯一候选时自动映射
    uniq = Lexicon([LexiconEntry(standard_name="性别", pinyin_initials="xb")])
    m2 = std.match_column(clean_column_name("XB"), uniq, {})
    assert m2.status == "auto" and m2.standard_name == "性别"


def test_boundary_same_standard_conflict_keeps_both_with_suffix():
    """同名冲突：白细胞 与 WBC 并存 -> 均保留 + 来源后缀 + conflicts 记录。"""
    df = _rng_df(["白细胞", "WBC"])
    res = standardize_dataframe(df)
    a, b = _by_raw(res, "白细胞"), _by_raw(res, "WBC")
    assert a.standard_name == b.standard_name == "白细胞计数"
    assert a.source_suffix == "[白细胞]" and b.source_suffix == "[WBC]"
    assert len(res.conflicts) == 1
    assert res.conflicts[0]["raw_columns"] == ["白细胞", "WBC"]
    assert list(df.columns) == ["白细胞", "WBC"]        # 数据列不合并


def test_boundary_ordinal_columns_not_merged():
    """序号列不合并：白细胞、白细胞_2 各占一行，数据原样。"""
    df = _rng_df(["白细胞", "白细胞_2"])
    assert len(df.columns) == 2
    res = standardize_dataframe(df)
    hits = [c for c in res.columns if c.standard_name == "白细胞计数"]
    assert len(hits) == 2
    assert {h.source_suffix for h in hits} == {"[白细胞]", "[白细胞_2]"}
    assert len(res.mapping_table) == 2


def test_boundary_all_unrecognized_still_works():
    """全部列未识别：流程继续，unknown_columns 全列，报告素材齐备。"""
    df = _rng_df(["QWER001", "ZXCV002", "JKLM003"])
    res = standardize_dataframe(df)
    assert res.unknown_columns == ["QWER001", "ZXCV002", "JKLM003"]
    assert res.pending_count == 0
    assert len(res.mapping_table) == 3
    assert all(c.status == "unknown" for c in res.columns)


def test_empty_dataframe_raises():
    with pytest.raises(DataShapeError):
        standardize_dataframe(pd.DataFrame())


def test_full_pinyin_fuzzy_recovers_typo():
    """全拼模糊：'血红蛋百'（错别字）-> 血红蛋白。"""
    df = _rng_df(["血红蛋百"])
    res = standardize_dataframe(df)
    assert _by_raw(res, "血红蛋百").standard_name == "血红蛋白"
    assert _by_raw(res, "血红蛋百").status == "auto"


# ------------------------------------------------------------------
# 用户级词典
# ------------------------------------------------------------------
def test_user_lexicon_wins_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(std, "_user_lexicon_path",
                        lambda tenant="default": str(tmp_path / f"{tenant}.json"))
    std.save_user_mapping("院内白细胞", "白细胞计数")
    assert std.load_user_lexicon()["院内白细胞"] == "白细胞计数"
    df = _rng_df(["院内白细胞"])
    res = std.standardize_dataframe(df)
    m = _by_raw(res, "院内白细胞")
    assert m.standard_name == "白细胞计数" and m.level == "L1"
    assert "用户词典" in m.note


def test_user_lexicon_does_not_touch_global(tmp_path, monkeypatch):
    before = load_global_lexicon()
    monkeypatch.setattr(std, "_user_lexicon_path",
                        lambda tenant="default": str(tmp_path / "default.json"))
    std.save_user_mapping("某院内编码XYZ", "白细胞计数")
    after = load_global_lexicon()
    assert before is after          # 全局词典缓存不变
    assert "某院内编码XYZ" not in after.by_alias


def test_reference_ranges_merged_into_lexicon():
    lex = load_global_lexicon()
    e = lex.by_standard["白细胞计数"]
    assert "wbc" in [x.lower() for x in e.english]          # REFERENCE_RANGES key 并入
    assert "白细胞计数" in e.aliases                          # label 并入
