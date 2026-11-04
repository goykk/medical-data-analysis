# medical-data-analysis/backend/tests/test_validators.py
"""医疗数据校验与变量类型推断的单元测试。"""

import numpy as np
import pandas as pd
import pytest

import validators
from config import ALPHA


def _type_of(schema, name):
    return next((v for v in schema if v["name"] == name), None)


def test_medical_dataset_passes_validation(medical_df):
    res = validators.validate_medical_dataset(medical_df)
    assert res["is_medical"] is True
    assert res["matched_fields"] >= 2
    assert res["match_score"] >= res["threshold"]["min_score"]
    assert set(res["options"]) == {"terminate", "continue"}


def test_non_medical_dataset_fails_validation(non_medical_df):
    res = validators.validate_medical_dataset(non_medical_df)
    assert res["is_medical"] is False
    assert res["options"] == ["terminate", "continue"]
    assert "可能不是医疗数据" in res["message"]


def test_glucose_not_suggested_as_outcome():
    """修复前子串匹配会让 'glucose' 命中 'os' 被误荐为结局；修复后按词边界匹配不再误伤。"""
    schema = [
        {"name": "glucose", "confirmed_type": "numeric", "excluded": False, "n_unique": 40},
        {"name": "dose", "confirmed_type": "numeric", "excluded": False, "n_unique": 10},
        {"name": "os", "confirmed_type": "categorical", "excluded": False, "n_unique": 2},
    ]
    cands = validators.suggest_outcome_candidates(
        pd.DataFrame({"glucose": [1], "dose": [1], "os": [1]}), schema)
    fields = {c["field"] for c in cands}
    assert "os" in fields
    assert "glucose" not in fields
    assert "dose" not in fields


def test_id_column_and_constant_column_excluded(medical_df):
    schema = validators.infer_variable_types(medical_df)
    pid = _type_of(schema, "患者ID")
    assert pid["excluded"] is True and "ID" in pid["exclude_reason"]
    center = _type_of(schema, "中心")
    assert center["excluded"] is True and "常量" in center["exclude_reason"]


def test_numeric_code_variable_is_categorical(medical_df):
    """数字编码分类变量（性别 0/1、吸烟 0/1）不得默认按数值型处理。"""
    schema = validators.infer_variable_types(medical_df)
    assert _type_of(schema, "性别")["inferred_type"] == "categorical"
    assert _type_of(schema, "吸烟")["inferred_type"] == "categorical"
    assert _type_of(schema, "结局")["inferred_type"] == "categorical"


def test_continuous_variable_is_numeric(medical_df):
    schema = validators.infer_variable_types(medical_df)
    assert _type_of(schema, "年龄")["inferred_type"] == "numeric"
    assert _type_of(schema, "体质指数")["inferred_type"] == "numeric"


def test_continuous_float_with_unique_values_not_id():
    """WDBC 类连续特征（取值近乎唯一但带小数）不得被误判为 ID 列。"""
    rng = np.random.default_rng(3)
    n = 569
    df = pd.DataFrame({
        "perimeter_mean": rng.normal(90, 20, n).round(4),   # 唯一值占比 >0.9 的浮点列
        "id": [f"{i:06d}" for i in range(n)],
    })
    schema = validators.infer_variable_types(df)
    peri = _type_of(schema, "perimeter_mean")
    assert peri["excluded"] is False and peri["inferred_type"] == "numeric"
    pid = _type_of(schema, "id")
    assert pid["excluded"] is True and "ID" in pid["exclude_reason"]


def test_integer_id_without_special_name_still_excluded():
    """名字不像编号但全整数且近乎唯一的列（如流水号 1001..）仍应判为 ID。"""
    n = 300
    df = pd.DataFrame({"就诊号": np.arange(1001, 1001 + n)})
    schema = validators.infer_variable_types(df)
    assert _type_of(schema, "就诊号")["excluded"] is True


def test_datetime_column_excluded():
    # 注意：行数需足够，否则唯一值占比 >0.9 的整数列会先命中 ID 规则（小样本的固有行为）
    df = pd.DataFrame({
        "入院日期": pd.to_datetime([f"2024-01-{i + 1:02d}" for i in range(10)]),
        "年龄": [50, 60, 70, 50, 60, 70, 55, 65, 50, 60],
    })
    schema = validators.infer_variable_types(df)
    dt = _type_of(schema, "入院日期")
    assert dt["inferred_type"] == "datetime" and dt["excluded"] is True
    assert _type_of(schema, "年龄")["excluded"] is False


def test_user_override_is_respected(medical_df):
    schema = validators.infer_variable_types(medical_df, overrides={"性别": "numeric"})
    item = _type_of(schema, "性别")
    assert item["confirmed_type"] == "numeric"
    assert item["inferred_type"] == "categorical"


def test_missing_rate_and_sample_values(medical_df):
    schema = validators.infer_variable_types(medical_df)
    glu = _type_of(schema, "空腹血糖")
    assert glu["missing_rate"] == pytest.approx(15 / len(medical_df), abs=1e-4)
    assert len(glu["sample_values"]) <= 3


def test_reference_range_attached_by_alias():
    schema = validators.infer_variable_types(pd.DataFrame({"空腹血糖": [5.0, 6.0, 7.0],
                                                           "bmi": [22.0, 23.0, 24.0]}))
    assert _type_of(schema, "空腹血糖")["reference_range"]["high"] == 6.1
    assert _type_of(schema, "bmi")["reference_range"]["low"] == 18.5


def test_outlier_detection_reports_reference_and_iqr():
    df = pd.DataFrame({"空腹血糖": [4.0, 5.0, 5.5, 6.0, 6.5, 33.3]})
    rep = validators.detect_medical_outliers(df, ["空腹血糖"])
    assert len(rep) == 1
    assert rep[0]["n_above_reference"] == 2        # 6.5 与 33.3 均超出参考区间上限 6.1
    assert rep[0]["n_below_reference"] == 0
    assert rep[0]["n_iqr_outlier"] >= 1


def test_outcome_candidates_sorted_and_excludes_id(medical_df):
    schema = validators.infer_variable_types(medical_df)
    cands = validators.suggest_outcome_candidates(medical_df, schema)
    names = [c["field"] for c in cands]
    assert "结局" in names
    assert "患者ID" not in names
    assert names[0] == "结局"


def test_outcome_candidates_skip_demographics_and_numeric_names():
    """人口学二值列与无表头产生的数字列名不得抢占结局候选。"""
    schema = [
        {"name": "SEX", "confirmed_type": "categorical", "excluded": False, "n_unique": 2},
        {"name": "Category", "confirmed_type": "categorical", "excluded": False, "n_unique": 4},
        {"name": "1.0000000e+000", "confirmed_type": "numeric", "excluded": False, "n_unique": 50},
        {"name": "Result", "confirmed_type": "categorical", "excluded": False, "n_unique": 3},
    ]
    df = pd.DataFrame({c["name"]: [1, 2] for c in schema})
    cands = validators.suggest_outcome_candidates(df, schema)
    fields = [c["field"] for c in cands]
    assert "SEX" not in fields
    assert "1.0000000e+000" not in fields
    assert fields[0] in ("Category", "Result")


def test_outcome_candidates_skip_age_and_pick_real_count_column():
    """Age/AGE 与性别种族一样属于人口学列；计数型结局（AIDS）仍应保留。"""
    schema = [
        {"name": "Sex", "confirmed_type": "categorical", "excluded": False, "n_unique": 2},
        {"name": "Age", "confirmed_type": "categorical", "excluded": False, "n_unique": 5},
        {"name": "Race", "confirmed_type": "categorical", "excluded": False, "n_unique": 5},
        {"name": "AIDS", "confirmed_type": "numeric", "excluded": False, "n_unique": 50},
        {"name": "Total", "confirmed_type": "numeric", "excluded": False, "n_unique": 50},
    ]
    df = pd.DataFrame({c["name"]: [1, 2] for c in schema})
    fields = [c["field"] for c in validators.suggest_outcome_candidates(df, schema)]
    assert "Age" not in fields and "Sex" not in fields and "Race" not in fields
    # Total 属于合计列（降权），AIDS 应排在它前面
    assert fields[0] == "AIDS"
    assert "Total" not in fields or fields.index("AIDS") < fields.index("Total")


def test_outcome_candidates_ignore_annotation_and_rater_columns():
    """评分者/受试者/实验条件列不能当结局；连续计数列（Count）应兜底为候选。"""
    schema = [
        {"name": "Automatic", "confirmed_type": "categorical", "excluded": False, "n_unique": 2},
        {"name": "Scorer_2", "confirmed_type": "categorical", "excluded": False, "n_unique": 2},
        {"name": "Subject", "confirmed_type": "categorical", "excluded": False, "n_unique": 3},
        {"name": "Count", "confirmed_type": "numeric", "excluded": False, "n_unique": 120},
    ]
    df = pd.DataFrame({c["name"]: [1, 2] for c in schema})
    fields = [c["field"] for c in validators.suggest_outcome_candidates(df, schema)]
    assert fields == ["Count"]


def test_outcome_candidates_feature_family_not_top():
    """宽表中同前缀特征族（FCFP6_1024_x）不得被当作结局，真实目标列（PXC50）应排第一。"""
    schema = [{"name": f"FCFP6_1024_{i}", "confirmed_type": "categorical",
               "excluded": False, "n_unique": 2} for i in range(60)]
    schema.append({"name": "MEDIAN_PXC50", "confirmed_type": "numeric",
                   "excluded": False, "n_unique": 60})
    df = pd.DataFrame({c["name"]: [1, 0] for c in schema})
    cands = validators.suggest_outcome_candidates(df, schema)
    fields = [c["field"] for c in cands]
    assert fields[0] == "MEDIAN_PXC50"
    assert not any(f.startswith("FCFP6_1024_") for f in fields)


def test_outcome_candidates_binaryclass_suffix_target_like():
    """binaryClass 这类后缀命名应视作目标列命名。"""
    schema = [
        {"name": "treat", "confirmed_type": "categorical", "excluded": False, "n_unique": 2},
        {"name": "binaryClass", "confirmed_type": "categorical", "excluded": False, "n_unique": 2},
    ]
    df = pd.DataFrame({c["name"]: [1, 2] for c in schema})
    fields = [c["field"] for c in validators.suggest_outcome_candidates(df, schema)]
    assert fields[0] == "binaryClass"


def test_id_detection_only_for_sequential_integer_codes():
    """连续编号 1..n 判为 ID；计数型整数列（每行都不同且不连续）不得被当成 ID 排除。"""
    n = 40
    df = pd.DataFrame({
        "visitseq": list(range(1, n + 1)),
        "cases": list(range(2, 202, 5)),
    })
    schema = validators.infer_variable_types(df)
    assert validators_test_excluded(schema, "visitseq") is True
    assert validators_test_excluded(schema, "cases") is False


def validators_test_excluded(schema, name):
    return next(v["excluded"] for v in schema if v["name"] == name)


def test_medical_validation_returns_matched_terms(medical_df):
    res = validators.validate_medical_dataset(medical_df)
    assert res["matched_terms"]
    assert "命中术语" in res["message"]


def test_medical_validation_omics_matrix():
    """GENE*X / g_*_at 这类组学矩阵应判为生物医学数据（rule=omics）。"""
    cols = {f"GENE{i}X": [0.1, 0.2, 0.3] for i in range(25)}
    df = pd.DataFrame({**cols, "class": ["a", "b", "a"]})
    res = validators.validate_medical_dataset(df)
    assert res["is_medical"] is True
    assert res["rule"] == "omics"
    assert res["omics_fields"] >= 25


def test_medical_validation_cardiology_abbreviations():
    """心脏病公开数据集常见缩写（cp/trestbps/thalach/oldpeak）应命中词库。"""
    df = pd.DataFrame({
        "age": [1, 2], "sex": [0, 1], "cp": [1, 2], "trestbps": [1, 2],
        "fbs": [0, 1], "thalach": [1, 2], "exang": [0, 1], "oldpeak": [1, 2],
        "num": [0, 1],
    })
    res = validators.validate_medical_dataset(df)
    assert res["is_medical"] is True
    assert res["matched_fields"] >= 3


def test_medical_validation_uses_filename_hint_and_terms():
    """字段命中不足时，文件名含医学词可作为补充证据，并返回命中的术语。"""
    df = pd.DataFrame({"glucose": [5.6, 7.1], "a": [1, 2], "b": [3, 4], "c": [5, 6],
                       "d": [7, 8], "e": [9, 10], "f": [11, 12], "g": [13, 14],
                       "h": [15, 16], "i": [17, 18]})
    res = validators.validate_medical_dataset(df, dataset_name="cardiac_study.csv")
    assert res["is_medical"] is True
    assert res["rule"] == "filename"
    assert "cardiac" in res["name_hint"]
    assert "glucose" in res["matched_terms"]
    assert "文件名包含医学提示词" in res["message"]


def test_strong_filename_allows_anonymized_medical_columns():
    """列名全部匿名化（V1..Vn）时，强医疗文件名必须仍能放行。

    回归：heart-switzerland / lung-cancer 这类公开数据集字段命中为 0，
    旧规则要求“文件名命中 + 至少 1 个字段命中”，导致真实医疗数据被误判。
    """
    df = pd.DataFrame({f"V{i}": [0, 1, 2, 3] for i in range(1, 13)})
    df["Class"] = [0, 1, 0, 1]
    res = validators.validate_medical_dataset(df, dataset_name="heart-switzerland.csv")
    assert res["matched_fields"] == 0
    assert res["is_medical"] is True
    assert res["rule"] == "filename"
    assert "heart" in res["name_hint"]

    res2 = validators.validate_medical_dataset(df, dataset_name="lung-cancer.csv")
    assert res2["is_medical"] is True
    assert res2["rule"] == "filename"


def test_generic_filename_does_not_override_zero_field_hits():
    """非医疗文件名不得仅凭“有个文件名”放行匿名化数据。"""
    df = pd.DataFrame({f"V{i}": [0, 1, 2, 3] for i in range(1, 13)})
    df["Class"] = [0, 1, 0, 1]
    res = validators.validate_medical_dataset(df, dataset_name="retail_sales.csv")
    assert res["matched_fields"] == 0
    assert res["is_medical"] is False
    assert res["rule"] == "none"


def test_generic_scene_filename_does_not_allow_zero_field_hits():
    """patient/hospital/medical/health 等通用场景词不能单独放行匿名化数据。"""
    df = pd.DataFrame({"sales": [1, 2, 3], "rating": [4, 5, 6],
                       "department": ["A", "B", "C"]})
    for name in ("health_insurance_claims.csv", "patient_satisfaction.csv",
                 "medical_devices_sales.csv", "hospitality.csv"):
        res = validators.validate_medical_dataset(df, dataset_name=name)
        assert res["is_medical"] is False, name
        assert res["rule"] == "none", name

    # 只要有一个真实医学字段命中，普通文件名提示仍然可以放行
    df2 = pd.DataFrame({"glucose": [5.6, 7.1], "sales": [1, 2],
                        "rating": [4, 5]})
    res2 = validators.validate_medical_dataset(df2, dataset_name="patient_satisfaction.csv")
    assert res2["is_medical"] is True
    assert res2["rule"] == "filename"


def test_alpha_constant():
    assert ALPHA == 0.05


def test_wdbc_like_dataset_passes_by_strong_clinical_field():
    """WDBC 这类字段名多为形态学测量的数据集，凭 diagnosis 字段应判定为医疗数据。"""
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "id": [f"{i:06d}" for i in range(60)],
        "radius_mean": rng.normal(14, 3, 60),
        "texture_mean": rng.normal(19, 4, 60),
        "perimeter_mean": rng.normal(92, 20, 60),
        "area_mean": rng.normal(650, 300, 60),
        "smoothness_mean": rng.normal(0.09, 0.01, 60),
        "diagnosis": rng.choice(["M", "B"], 60),
    })
    res = validators.validate_medical_dataset(df)
    assert res["is_medical"] is True
    assert res["rule"] == "strong_clinical"
    assert "diagnosis" in res["strong_clinical_fields"]


def test_non_medical_has_no_strong_clinical_field(non_medical_df):
    assert validators.strong_clinical_fields(non_medical_df) == []
    assert validators.validate_medical_dataset(non_medical_df)["rule"] == "none"
