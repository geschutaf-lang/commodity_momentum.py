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

# ── CSS 디자인 ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.title-block { border-left: 4px solid #f59e0b; padding: 0.4rem 1rem; margin-bottom: 1.5rem; }
.title-block h1 { font-size: 1.6rem; font-weight: 600; margin: 0; color: #78350f; }
.metric-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 1rem; text-align: center; }
.winner-card { background: #fdf4ff; border: 2px solid #d946ef; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
.winner-card .ticker { font-size: 2.5rem; font-weight: 600; color: #86198f; font-family: 'IBM Plex Mono', monospace; }
.tag-pass { background: #dcfce7; color: #15803d; padding: 2px 10px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
  <h1>PTP-Free Commodity Momentum</h1>
  <p>한국인 투자자를 위한 세금(PTP 10%) 원천 차단형 원자재/관련 기업 모멘텀 분석기</p>
</div>
""", unsafe_allow_html=True)

# ── 원자재 ETF 핵심 라인업 (PTP 면제 100% 안전 종목 11개) ────────
COMMODITY_ETFS = {
    # 종합 원자재 (No K-1, 세금 우회 구조)
    'PDBC': '종합 원자재 최적화 (PDBC)',
    'COMB': '종합 원자재 No K-1 (COMB)',
    
    # 귀금속 (실물 보유로 PTP 완전 면제)
    'GLD': '금 실물 (GLD)',
    'SLV': '은 실물 (SLV)',
    'PALL': '팔라듐 실물 (PALL)',
    'PPLT': '백금 실물 (PPLT)',
    
    # 에너지/산업금속/농산물 
    # (선물 ETF 대신 원자재를 캐내는 '생산 기업' 주식으로 우회하여 PTP 원천 차단)
    'XLE': '에너지 생산 기업 (XLE)',
    'COPX': '구리 광산 기업 (COPX)',
    'URA': '우라늄 채굴 기업 (URA)',
    'MOO': '글로벌 농업/식량 기업 (MOO)',
    'GDX': '글로벌 금광 기업 (GDX)'
}

# ── 모멘텀 계산 함수 ──────────────────────────────────────────
def avg_momentum(series):
    s = series.dropna()
    if len(s) < 13: return np.nan, np.nan, np.nan, np.nan, np.nan
    p = s.iloc[-1]
    r1, r3, r6, r12 = p/s.iloc[-2]-1, p/s.iloc[-4]-1, p/s.iloc[-7]-1, p/s.iloc[-13]-1
    return (r1+r3+r6+r12)/4, r1, r3, r6, r12

# ── 실행 로직 ─────────────────────────────────────────────────
if st.button("🚀 안전한 원자재 모멘텀 분석 시작", type="primary"):
    with st.status("PTP 면제 원자재 데이터 수집 및 분석 중...", expanded=True) as status:
        tickers = list(COMMODITY_ETFS.keys())
        st.write(f"📡 {len(tickers)}개 안전 종목 데이터를 수집합니다...")
        
        end = datetime.today()
        start = end - timedelta(days=430)
        
        # 주가 다운로드 (TIP 필터 포함)
        raw = yf.download(tickers + ['TIP'], start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True, progress=False)
        prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
        monthly = prices.resample('ME').last()
        
        # TIP 필터 계산
        tip_avg, _, _, _, _ = avg_momentum(monthly['TIP'])
        tip_pass = not np.isnan(tip_avg) and tip_avg > 0
        
        rows = []
        for tk in tickers:
            if tk in monthly.columns:
                m, r1, r3, r6, r12 = avg_momentum(monthly[tk])
                if not np.isnan(m):
                    rows.append({
                        '티커': tk, 
                        '원자재/기업 섹터': COMMODITY_ETFS[tk], 
                        '평균모멘텀(%)': round(m*100, 2), 
                        '1M': round(r1*100, 2), 
                        '3M': round(r3*100, 2), 
                        '6M': round(r6*100, 2), 
                        '12M': round(r12*100, 2)
                    })
        
        df = pd.DataFrame(rows).sort_values('평균모멘텀(%)', ascending=False).reset_index(drop=True)
        df.index += 1
        status.update(label="안전 종목 분석 완료!", state="complete")

    st.divider()
    
    # 상단 정보 카드
    c1, c2, c3 = st.columns(3)
    tip_txt = f"{tip_avg*100:+.2f}%" if not np.isnan(tip_avg) else "N/A"
    c1.markdown(f'<div class="metric-box"><div style="font-size:0.75rem; color:#92400e;">글로벌 거시 필터(TIP)</div><div style="font-size:1.5rem; font-weight:600;">{tip_txt}</div><div style="margin-top:5px;">{"<span class='tag-pass'>PASS</span>" if tip_pass else "<span style='background:#fee2e2; color:#b91c1c; padding:2px 10px; border-radius:99px; font-size:0.8rem; font-weight:600;'>BLOCK</span>"}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div style="font-size:0.75rem; color:#92400e;">PTP 면제 안전 종목</div><div style="font-size:1.5rem; font-weight:600;">{len(df)}개</div><div style="margin-top:5px; font-size:0.8rem; color:#b45309;">실물자산 및 생산기업</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div style="font-size:0.75rem; color:#92400e;">세금 리스크</div><div style="font-size:1.5rem; font-weight:600;">0% (Zero)</div><div style="margin-top:5px; font-size:0.8rem; color:#b45309;">원천징수 우려 없음</div></div>', unsafe_allow_html=True)

    if not tip_pass:
        st.error("⚠️ TIP 필터 차단: 시장 유동성이 축소되고 있습니다. 변동성이 큰 원자재 투자를 보류하고 현금(달러)을 보유하세요.")
    elif not df.empty:
        best = df.iloc[0]
        if best['평균모멘텀(%)'] <= 0:
            st.warning(f"⚠️ 현재 1위인 {best['원자재/기업 섹터']}조차 추세가 꺾였습니다({best['평균모멘텀(%)']:+.2f}%). 원자재 투자를 쉬어가는 것이 좋습니다.")
        else:
            st.markdown(f"""
            <div class="winner-card">
              <div style="font-size:0.8rem; color:#86198f; text-transform:uppercase; letter-spacing:0.1em;">이번 달 추천 종목 (PTP 안전)</div>
              <div class="ticker">{best['티커']}</div>
              <div class="name">{best['원자재/기업 섹터']}</div>
              <div style="font-size:1.3rem; font-weight:600; color:#a21caf; margin-top:10px;">평균 상승 추세: {best['평균모멘텀(%)']:+.2f}%</div>
              <div style="font-size:0.85rem; color:#c026d3; margin-top:8px;">1M: {best['1M']}% | 3M: {best['3M']}% | 6M: {best['6M']}% | 12M: {best['12M']}%</div>
            </div>""", unsafe_allow_html=True)
        
        st.subheader("🏆 PTP 면제 원자재/관련 기업 순위표")
        st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['평균모멘텀(%)']), use_container_width=True, height=500)
