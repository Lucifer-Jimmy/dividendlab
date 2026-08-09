#!/usr/bin/env python3
"""数据源假设校验脚本。

docs/data-requirements.md 中的字段规格建立在若干可被上游静默改变的假设上
（分红单位、除息日超前、hfq 是否含分红等）。本脚本重跑这些校验，
在 akshare 或东财改版导致假设失效时给出明确失败，而不是让错误数据静默流入。

用法：uv run python scripts/verify_data_source.py

注意：依赖外部网络，东财接口不稳定属常态，失败会重试后标记 SKIP 而非 FAIL。
"""

from __future__ import annotations

import random
import sys
import time
from datetime import date

import akshare as ak
import pandas as pd

# 实测基准（2026-08-08）。若上游口径变化，这些断言应当失败。
DIV_UNIT_CASE = {"code": "510880", "ex_date": "2026-01-21", "declared": 0.143}
KNOWN_DIVIDEND_ETFS = ["510880", "515180", "563020", "159545", "515080"]
MONTHLY_DIVIDEND_ETF = "510720"  # 实测连续 27 个月分红

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    print(f"[{status}] {name}: {detail}")


def retry(fn, tries: int = 4, **kwargs):
    """东财接口不稳定，重试后仍失败则返回 None（记 SKIP，不算 FAIL）。"""
    for i in range(tries):
        try:
            return fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 网络异常类型不可穷举
            if i == tries - 1:
                print(f"  give up after {tries} tries: {type(exc).__name__}")
                return None
            time.sleep(3 * (i + 1) + random.uniform(0, 2))
    return None


def load_dividends(years: list[str]) -> pd.DataFrame | None:
    """fund_fh_em 的 year 默认值硬编码为 '2025'，必须显式按年遍历。"""
    frames = []
    for year in years:
        df = retry(ak.fund_fh_em, year=year)
        if df is None:
            continue
        frames.append(df)
        time.sleep(random.uniform(3, 5))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["基金代码"] = out["基金代码"].astype(str).str.zfill(6)
    for col in ("权益登记日", "除息日期", "分红发放日"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def check_etf_coverage(div: pd.DataFrame) -> None:
    """假设：fund_fh_em 收录场内 ETF（调研阶段曾存疑，实测确认收录）。"""
    codes = set(div["基金代码"])
    missing = [c for c in KNOWN_DIVIDEND_ETFS if c not in codes]
    if missing:
        record("ETF 收录", "FAIL", f"以下已知分红 ETF 未出现在 fund_fh_em: {missing}")
    else:
        record("ETF 收录", "PASS", f"{len(KNOWN_DIVIDEND_ETFS)} 只已知红利 ETF 全部命中")


def check_monthly_dividend(div: pd.DataFrame) -> None:
    """假设：存在真正月月分红的 ETF，且除息日可用于逐月判定。"""
    sub = div[div["基金代码"] == MONTHLY_DIVIDEND_ETF]
    months = sorted({p.strftime("%Y-%m") for p in sub["除息日期"].dropna()})
    if len(months) < 12:
        record("月月分红样本", "FAIL", f"{MONTHLY_DIVIDEND_ETF} 仅覆盖 {len(months)} 个月，预期 >=12")
    else:
        record("月月分红样本", "PASS", f"{MONTHLY_DIVIDEND_ETF} 覆盖 {len(months)} 个月 ({months[0]}~{months[-1]})")


def check_future_records(div: pd.DataFrame) -> None:
    """假设：接口含「已公告未除息」的未来记录 —— 这是前视偏差的主要来源。

    若某天该假设不再成立也只是提示，真正危险的是反向情况（有未来数据却未过滤）。
    """
    today = pd.Timestamp(date.today())
    future = div[div["除息日期"] > today]
    if future.empty:
        record("未来分红记录", "INFO", "当前无未来除息记录；ex_date <= 基准日 的过滤仍必须保留")
    else:
        latest = future["除息日期"].max().date()
        record(
            "未来分红记录",
            "PASS",
            f"存在 {len(future)} 条未来除息记录（最远 {latest}）→ 必须按 ex_date <= 基准日过滤",
        )


def check_dividend_unit() -> None:
    """假设：分红单位为元/份。用除息日价格跳空交叉验证，而非信任字段说明。

    若单位实为「每10份」，跳空幅度将只有声明值的约 1/10。
    """
    case = DIV_UNIT_CASE
    ex = pd.Timestamp(case["ex_date"])
    hist = retry(
        ak.fund_etf_hist_em,
        symbol=case["code"],
        period="daily",
        start_date=(ex - pd.Timedelta(days=7)).strftime("%Y%m%d"),
        end_date=(ex + pd.Timedelta(days=3)).strftime("%Y%m%d"),
        adjust="",
    )
    if hist is None:
        record("分红单位", "SKIP", "行情接口不可用（东财网络问题）")
        return

    hist["日期"] = pd.to_datetime(hist["日期"])
    hist = hist.sort_values("日期").reset_index(drop=True)
    hist["前收"] = hist["收盘"].shift(1)

    row = hist[hist["日期"] == ex]
    if row.empty:
        record("分红单位", "SKIP", f"{case['ex_date']} 无行情数据")
        return

    gap = float(row["前收"].iloc[0] - row["开盘"].iloc[0])
    declared = case["declared"]
    # 跳空同时受当日行情影响，只要求量级吻合（排除 10 倍口径差异）
    if abs(gap - declared) < declared * 0.5:
        record("分红单位", "PASS", f"跳空 {gap:.4f} vs 声明 {declared} → 元/份")
    elif abs(gap - declared / 10) < declared * 0.05:
        record("分红单位", "FAIL", f"跳空 {gap:.4f} 接近声明值 1/10 → 疑似口径已变为每10份")
    else:
        record("分红单位", "WARN", f"跳空 {gap:.4f} 与声明 {declared} 不吻合，需人工确认")


def check_hfq_includes_dividend() -> None:
    """假设：hfq 后复权已处理分红。用于校验自建总收益序列，不作为落库口径。"""
    case = DIV_UNIT_CASE
    ex = pd.Timestamp(case["ex_date"])
    start = (ex - pd.Timedelta(days=4)).strftime("%Y%m%d")
    end = (ex + pd.Timedelta(days=2)).strftime("%Y%m%d")

    raw = retry(ak.fund_etf_hist_em, symbol=case["code"], period="daily",
                start_date=start, end_date=end, adjust="")
    time.sleep(random.uniform(3, 5))
    adj = retry(ak.fund_etf_hist_em, symbol=case["code"], period="daily",
                start_date=start, end_date=end, adjust="hfq")
    if raw is None or adj is None:
        record("hfq 含分红", "SKIP", "行情接口不可用（东财网络问题）")
        return

    def pct_on_ex(df: pd.DataFrame) -> float | None:
        df["日期"] = pd.to_datetime(df["日期"])
        row = df[df["日期"] == ex]
        return None if row.empty else float(row["涨跌幅"].iloc[0])

    raw_pct, adj_pct = pct_on_ex(raw), pct_on_ex(adj)
    if raw_pct is None or adj_pct is None:
        record("hfq 含分红", "SKIP", f"{case['ex_date']} 无行情数据")
    elif abs(adj_pct) < abs(raw_pct) / 2:
        record("hfq 含分红", "PASS", f"除息日 不复权 {raw_pct}% vs 后复权 {adj_pct}% → hfq 已还原分红")
    else:
        record("hfq 含分红", "FAIL", f"除息日 不复权 {raw_pct}% vs 后复权 {adj_pct}% → hfq 疑似未处理分红")


def check_index_weight() -> None:
    """假设：中证权重接口可用，仅返回最新截面，且含成分券代码可 join 行业。"""
    w = retry(ak.index_stock_cons_weight_csindex, symbol="000922")
    if w is None:
        record("指数权重", "SKIP", "中证接口不可用")
        return

    total = float(w["权重"].sum())
    as_of = sorted({str(d) for d in w["日期"].unique()})
    if abs(total - 100) > 1:
        record("指数权重", "FAIL", f"权重和 {total:.3f} 偏离 100")
    elif len(as_of) > 1:
        record("指数权重", "WARN", f"返回多个截面日期 {as_of}，与「仅最新一期」假设不符")
    else:
        record("指数权重", "PASS", f"000922 {len(w)} 只成分，权重和 {total:.3f}，截面 {as_of[0]}")

    if "成分券代码" not in w.columns:
        record("成分券代码", "FAIL", "缺少成分券代码列，无法 join 行业分类")


def main() -> int:
    print("=== DividendLab 数据源假设校验 ===\n")

    this_year = date.today().year
    div = load_dividends([str(this_year - 1), str(this_year)])
    if div is None:
        record("分红接口", "SKIP", "fund_fh_em 不可用，跳过全部分红校验")
    else:
        record("分红接口", "PASS", f"取得 {len(div)} 条记录")
        check_etf_coverage(div)
        check_monthly_dividend(div)
        check_future_records(div)

    check_dividend_unit()
    time.sleep(random.uniform(3, 5))
    check_hfq_includes_dividend()
    time.sleep(random.uniform(3, 5))
    check_index_weight()

    failed = [r for r in results if r[1] == "FAIL"]
    skipped = [r for r in results if r[1] == "SKIP"]
    print(f"\n=== 汇总：{len(results)} 项，FAIL {len(failed)}，SKIP {len(skipped)} ===")
    if failed:
        print("失败项意味着 docs/data-requirements.md 的假设已失效，需复核后更新文档：")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
