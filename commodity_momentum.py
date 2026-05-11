# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PTP-Free Commodity Momentum",
    page_icon="🛢️",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.title-block { border-left: 4px solid #f59e0b; padding: 0.4rem 1rem; margin-bottom: 1.5rem; }
.title-block h1 { font-size: 1.6rem; font-weight: 600; margin: 0; color: #78350f !important; }
.metric-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
              padding: 1rem; text-align: center; color: #1c1917 !important; }
.winner-card { background: #fdf4ff; border: 2px solid #d946ef; border-radius: 12px;
               padding: 1.5rem; margin: 1rem 0; color: #1c1917 !important; }
.winner-card .ticker { font-size: 2.5rem; font-weight: 600; color: #86198f !important;
                       font-family: 'IBM Plex Mono', monospace; }
.winner-card .name   { color: #4a044e !important; }
.tag-pass { background: #dcfce7; color: #15803d !important; padding: 2px 10px;
            border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
.tag-block { background: #fee2e2; color: #b91c1c !important; padding: 2px 10px;
             border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
  <h1>PTP-Free Commodity Momentum</h1>
  <p>한국인 투자자를 위한 세금(PTP 10%) 원천 차단형 원자재/관련 기업 모멘텀 분석기</p>
</div>
""", unsafe_allow_html=True)

# ── PTP 면제 안전 종목 (중복 없이 정의) ──────────────────────────────────────
COMMODITY_ETFS = {
    # 종합 원자재 (No K-1 구조)
    'PDBC': '종합 원자재 최적화 (PDBC)',
    'COMB': '종합 원자재 No K-1 (COMB)',
    # 귀금속 실물 (PTP 완전 면제)
    'GLD':  '금 실물 (GLD)',
    'SLV':  '은 실물 (SLV)',
    'PALL': '팔라듐 실물 (PALL)',
    'PPLT': '백금 실물 (PPLT)',
    # 원자재 생산 기업 주식 (PTP 원천 차단)
    'XLE':  '에너지 생산 기업 (XLE)',
    'COPX': '구리 광산 기업 (COPX)',
    'URA':  '우라늄 채굴 기업 (URA)',
    'MOO':  '글로벌 농업/식량 기업 (MOO)',
    'GDX':  '글로벌 금광 기업 (GDX)',
}

# ── [BUG FIX 1] 날짜 기반 모멘텀 계산 ────────────────────────────────────────
# 기존: s.iloc[-2], s.iloc[-4] 등 인덱스 기준 → 데이터 공백 달 발생 시 오차
# 수정: pd.DateOffset으로 정확한 n개월 전 날짜를 찾아 nearest 매칭
def get_price_n_months_ago(series: pd.Series, n: int) -> float | None:
    """월말 시리즈에서 정확히 n개월 전에 가장 가까운 가격을 반환."""
    if series.empty:
        return None
    target = series.index[-1] - pd.DateOffset(months=n)
    # 가장 가까운 인덱스 위치
    loc = series.index.get_indexer([target], method='nearest')[0]
    if loc < 0:
        return None
    # 너무 멀리 떨어진 경우(45일 이상) 무효 처리
    actual = series.index[loc]
    if abs((actual - target).days) > 45:
        return None
    return series.iloc[loc]


def avg_momentum(series: pd.Series) -> tuple:
    """
    Returns (avg, r1, r3, r6, r12) 모두 소수(0.05 = 5%).
    데이터 부족 시 전체 nan 반환.
    """
    s = series.dropna()
    if len(s) < 5:          # 최소 5개 월말 데이터 필요
        return (np.nan,) * 5

    p = s.iloc[-1]          # 최신 월말 가격

    p1m  = get_price_n_months_ago(s, 1)
    p3m  = get_price_n_months_ago(s, 3)
    p6m  = get_price_n_months_ago(s, 6)
    p12m = get_price_n_months_ago(s, 12)

    def ret(base):
        return (p - base) / base if (base and base != 0) else np.nan

    r1, r3, r6, r12 = ret(p1m), ret(p3m), ret(p6m), ret(p12m)

    vals = [v for v in (r1, r3, r6, r12) if not np.isnan(v)]
    avg  = np.mean(vals) if vals else np.nan

    return avg, r1, r3, r6, r12


# ── [BUG FIX 2] yfinance MultiIndex 안전 파싱 ────────────────────────────────
# 기존: isinstance 분기만으로 처리 → 단일 티커일 때 열 이름 불일치 가능
# 수정: 항상 MultiIndex 기준으로 'Close' 레벨만 추출하고 컬럼명 정규화
def download_close(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by='ticker',   # 명시적으로 ticker 기준 그룹화
    )
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # (Price, Ticker) 형태 → Close 레벨만 추출
        if 'Close' in raw.columns.get_level_values(0):
            close = raw['Close']
        elif 'Close' in raw.columns.get_level_values(1):
            close = raw.xs('Close', axis=1, level=1)
        else:
            close = raw.iloc[:, raw.columns.get_level_values(1) == 'Close']
    else:
        close = raw[['Close']] if 'Close' in raw.columns else raw

    # 컬럼이 단일 티커 문자열일 수 있으므로 list로 보장
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    return close


# ── 메인 실행 ─────────────────────────────────────────────────────────────────
if st.button("🚀 안전한 원자재 모멘텀 분석 시작", type="primary"):

    tickers = list(COMMODITY_ETFS.keys())
    all_tickers = tickers + ['TIP']

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=430)   # 14개월치 확보

    with st.status("PTP 면제 원자재 데이터 수집 및 분석 중...", expanded=True) as status:
        st.write(f"📡 {len(tickers)}개 안전 종목 + TIP 필터 다운로드 중...")

        prices  = download_close(all_tickers, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))

        if prices.empty:
            st.error("데이터를 불러오지 못했습니다. 네트워크 상태를 확인하세요.")
            st.stop()

        # 월말 리샘플 (ME = Month End)
        monthly = prices.resample('ME').last()

        # ── TIP 필터 ──────────────────────────────────────────────────────────
        if 'TIP' not in monthly.columns:
            st.warning("TIP 데이터를 가져오지 못해 필터를 건너뜁니다.")
            tip_avg, tip_pass = np.nan, True
        else:
            tip_avg, *_ = avg_momentum(monthly['TIP'])
            tip_pass    = (not np.isnan(tip_avg)) and (tip_avg > 0)

        # ── 각 ETF 모멘텀 계산 ───────────────────────────────────────────────
        rows = []
        for tk in tickers:
            if tk not in monthly.columns:
                st.warning(f"⚠️ {tk} 데이터 없음 — 건너뜁니다.")
                continue
            m, r1, r3, r6, r12 = avg_momentum(monthly[tk])
            if np.isnan(m):
                st.warning(f"⚠️ {tk} 데이터 부족 — 건너뜁니다.")
                continue
            rows.append({
                '티커':             tk,
                '원자재/기업 섹터': COMMODITY_ETFS[tk],
                '평균모멘텀(%)':    round(m   * 100, 2),
                '1M(%)':           round(r1  * 100, 2) if not np.isnan(r1)  else None,
                '3M(%)':           round(r3  * 100, 2) if not np.isnan(r3)  else None,
                '6M(%)':           round(r6  * 100, 2) if not np.isnan(r6)  else None,
                '12M(%)':          round(r12 * 100, 2) if not np.isnan(r12) else None,
            })

        if not rows:
            st.error("유효한 종목이 없습니다.")
            st.stop()

        df = (pd.DataFrame(rows)
                .sort_values('평균모멘텀(%)', ascending=False)
                .reset_index(drop=True))
        df.index += 1

        status.update(label="✅ 분석 완료!", state="complete")

    # ── 요약 지표 카드 ────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)

    tip_txt  = f"{tip_avg*100:+.2f}%" if not np.isnan(tip_avg) else "N/A"
    tip_html = '<span class="tag-pass">PASS ✅</span>' if tip_pass else '<span class="tag-block">BLOCK ❌</span>'

    c1.markdown(f"""
    <div class="metric-box">
      <div style="font-size:0.75rem; color:#92400e;">글로벌 거시 필터 (TIP)</div>
      <div style="font-size:1.5rem; font-weight:600;">{tip_txt}</div>
      <div style="margin-top:5px;">{tip_html}</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-box">
      <div style="font-size:0.75rem; color:#92400e;">PTP 면제 안전 종목</div>
      <div style="font-size:1.5rem; font-weight:600;">{len(df)}개</div>
      <div style="margin-top:5px; font-size:0.8rem; color:#b45309;">실물자산 및 생산기업</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-box">
      <div style="font-size:0.75rem; color:#92400e;">세금 리스크</div>
      <div style="font-size:1.5rem; font-weight:600;">0% (Zero)</div>
      <div style="margin-top:5px; font-size:0.8rem; color:#b45309;">원천징수 우려 없음</div>
    </div>""", unsafe_allow_html=True)

    # ── TIP 차단 시 조기 종료 ─────────────────────────────────────────────────
    if not tip_pass:
        st.error(
            "⚠️ TIP 필터 차단: 물가 기대가 꺾이고 있습니다. "
            "원자재 투자를 보류하고 현금(달러 MMF 등)을 보유하세요."
        )
        st.stop()

    # ── 위너 카드 ─────────────────────────────────────────────────────────────
    best = df.iloc[0]

    if best['평균모멘텀(%)'] <= 0:
        st.warning(
            f"⚠️ 1위 {best['원자재/기업 섹터']}조차 추세가 음수입니다 "
            f"({best['평균모멘텀(%)']:+.2f}%). 원자재 투자를 쉬어가는 것이 좋습니다."
        )
    else:
        # None 값 포맷 처리
        def fmt(v):
            return f"{v:+.2f}%" if v is not None else "N/A"

        st.markdown(f"""
        <div class="winner-card">
          <div style="font-size:0.8rem; color:#86198f; text-transform:uppercase;
                      letter-spacing:0.1em;">이번 달 추천 종목 (PTP 안전)</div>
          <div class="ticker">{best['티커']}</div>
          <div class="name">{best['원자재/기업 섹터']}</div>
          <div style="font-size:1.3rem; font-weight:600; color:#a21caf; margin-top:10px;">
            평균 상승 추세: {best['평균모멘텀(%)']:+.2f}%
          </div>
          <div style="font-size:0.85rem; color:#c026d3; margin-top:8px;">
            1M: {fmt(best['1M(%)'])} &nbsp;|&nbsp;
            3M: {fmt(best['3M(%)'])} &nbsp;|&nbsp;
            6M: {fmt(best['6M(%)'])} &nbsp;|&nbsp;
           12M: {fmt(best['12M(%)'])}
          </div>
        </div>""", unsafe_allow_html=True)

    # ── 순위표 ────────────────────────────────────────────────────────────────
    st.subheader("🏆 PTP 면제 원자재/관련 기업 순위표")

    # [BUG FIX 3] subset 컬럼명을 실제 df 컬럼명과 일치시킴 (원본 코드 잘림 버그 수정)
    numeric_cols = ['평균모멘텀(%)', '1M(%)', '3M(%)', '6M(%)', '12M(%)']
    display_df   = df.copy()
    for col in numeric_cols:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

    st.dataframe(
        display_df.style
            .background_gradient(cmap='RdYlGn', subset=numeric_cols)
            .format({c: "{:+.2f}%" for c in numeric_cols}, na_rep="N/A"),
        use_container_width=True,
        height=420,
    )

    st.caption(
        "📌 매월 말일 기준으로 평균 모멘텀 1위 ETF 1개를 매수하는 단일 모멘텀 전략입니다. "
        "과거 수익률이 미래를 보장하지 않습니다."
    )
