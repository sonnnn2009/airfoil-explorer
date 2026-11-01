"""익형 설계 탐색기 (정보 과제 연구: 익형 형상 데이터 기반 공력 대리모델과 역설계)

실행:  streamlit run app.py
필요한 데이터(data/ 폴더): labels_slim.csv.gz, features.csv, coords_resampled.npz,
                          q3_grid_xfoil.csv, q4_final_designs.csv (선택: q4_results.csv)
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
X_COLS = ['t_max', 'x_t', 'camber_max', 'x_c', 'alpha', 'logRe']
ALPHA_CURVE = np.arange(-5, 15.01, 0.5)      # 그래프용 받음각
ALPHA_LD = np.arange(0, 12.01, 0.5)          # 최대 L/D 계산용 (실속 이후 제외, 탐구와 같은 기준)
RE_OPTIONS = {'1×10⁵': 100_000, '2×10⁵': 200_000, '5×10⁵': 500_000}
PINN = dict(t=0.287, m=0.028, p=0.4)
trapz = getattr(np, 'trapezoid', None) or np.trapz
XF = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 1001)))

st.set_page_config(page_title='익형 설계 탐색기', page_icon='✈️', layout='wide')


# ---------------- 데이터와 모델 ----------------
@st.cache_data(show_spinner=False)
def load_data():
    lab = pd.read_csv(f'{D}/labels_slim.csv.gz')
    lab['logRe'] = np.log10(lab.Re)
    lab['logCD'] = np.log10(lab.CD)
    feat = pd.read_csv(f'{D}/features.csv')
    npz = np.load(f'{D}/coords_resampled.npz', allow_pickle=True)
    shapes = dict(zip(npz['files'], npz['shapes']))
    grid = pd.read_csv(f'{D}/q3_grid_xfoil.csv')
    final = pd.read_csv(f'{D}/q4_final_designs.csv') if os.path.exists(f'{D}/q4_final_designs.csv') else None
    q4 = pd.read_csv(f'{D}/q4_results.csv') if os.path.exists(f'{D}/q4_results.csv') else None
    return lab, feat, npz['x'], shapes, grid, final, q4


@st.cache_resource(show_spinner=False)
def train_models(_lab):
    """웹 도구용 경량 랜덤포레스트 (메모리 절약을 위해 노트북 모델보다 나무 수와 잎 크기를 조정)"""
    kw = dict(n_estimators=80, min_samples_leaf=8, max_samples=0.5, n_jobs=-1, random_state=0)
    rf_cl = RandomForestRegressor(**kw).fit(_lab[X_COLS], _lab.CL)
    rf_cd = RandomForestRegressor(**kw).fit(_lab[X_COLS], _lab.logCD)
    return rf_cl, rf_cd


# ---------------- 형상 함수 (노트북과 같은 정의) ----------------
def naca4(m, p, t, n=101):
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m == 0:
        yc = dy = np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dy = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    th = np.arctan(dy)
    up = np.c_[x - yt * np.sin(th), yc + yt * np.cos(th)]
    lo = np.c_[x + yt * np.sin(th), yc - yt * np.cos(th)]
    return up, lo


def features_from_surfaces(xu, yu, xl, yl):
    yu = np.interp(XF, xu, yu); yl = np.interp(XF, xl, yl)
    t, cam = yu - yl, 0.5 * (yu + yl)
    i_t, i_c = int(np.argmax(t)), int(np.argmax(np.abs(cam)))
    area = trapz(t, XF)
    y_bar = trapz(0.5 * (yu**2 - yl**2), XF) / area
    I = trapz((yu**3 - yl**3) / 3, XF) - area * y_bar**2
    return dict(t_max=t[i_t], x_t=XF[i_t], camber_max=cam[i_c], x_c=XF[i_c], I=I)


def naca_features(t, m, p):
    up, lo = naca4(m, p, t)
    return features_from_surfaces(up[:, 0], up[:, 1], lo[:, 0], lo[:, 1])


def predict(models, feats, re, alphas):
    rf_cl, rf_cd = models
    X = pd.DataFrame([{**{k: feats[k] for k in X_COLS[:4]}, 'alpha': a, 'logRe': np.log10(re)} for a in alphas])
    cl = rf_cl.predict(X[X_COLS]); cd = 10 ** rf_cd.predict(X[X_COLS])
    return cl, cd


def surrogate_ldmax_batch(models, pop, re):
    rf_cl, rf_cd = models
    F = pd.DataFrame([naca_features(t, m, p) for t, m, p in pop])
    rep = F.loc[F.index.repeat(len(ALPHA_LD))].reset_index(drop=True)
    rep['alpha'] = np.tile(ALPHA_LD, len(F)); rep['logRe'] = np.log10(re)
    ld = (rf_cl.predict(rep[X_COLS]) / 10 ** rf_cd.predict(rep[X_COLS])).reshape(len(F), len(ALPHA_LD))
    return F.I.values, ld.max(1)


def pareto(df, x, y):
    v = df[[x, y]].values
    keep = [not np.any((v[:, 0] >= a) & (v[:, 1] >= b) & ((v[:, 0] > a) | (v[:, 1] > b))) for a, b in v]
    return df[keep].sort_values(x)


def dat_text(t, m, p, name):
    up, lo = naca4(m, p, t)
    xy = np.r_[up[::-1], lo[1:]]
    return name + '\n' + ''.join(f'{x:.6f} {y:.6f}\n' for x, y in xy)


# ---------------- 시작 ----------------
st.title('✈️ 익형 설계 탐색기')
st.caption('UIUC 익형 1,600여 개와 XFOIL 계산값으로 학습한 대리모델(랜덤포레스트)로 형상의 양력·항력을 예측하고, '
           '강성(단면 이차모멘트)과 공력 효율(최대 양항비)을 함께 비교합니다.')

with st.spinner('데이터를 불러오고 대리모델을 학습하는 중입니다 (처음 한 번만, 1~2분 걸릴 수 있음)...'):
    lab, feat, XS, shapes, grid, final, q4 = load_data()
    models = train_models(lab)

# 탐구에서 측정한 대리모델 신뢰도 (XFOIL 직접 계산값과 비교, Re 2×10^5)
gv = grid.iloc[:-1].dropna(subset=['XFOIL_200k'])
err_all = ((gv.LDmax_200k - gv.XFOIL_200k).abs() / gv.XFOIL_200k).mean() * 100
I_pinn = grid.iloc[-1].I
thick = gv[gv.I >= I_pinn]
rho_thick = spearmanr(thick.LDmax_200k, thick.XFOIL_200k).correlation

# ---------------- 입력 (사이드바) ----------------
with st.sidebar:
    st.header('형상 입력')
    mode = st.radio('입력 방식', ['NACA 4자리 형상 직접 설계', 'UIUC 실제 익형 선택'])
    if mode.startswith('NACA'):
        t = st.slider('최대 두께비 t/c', 0.06, 0.30, 0.12, 0.005)
        m = st.slider('최대 캠버 m', 0.0, 0.06, 0.02, 0.001)
        p = st.slider('최대 캠버 위치 p', 0.2, 0.6, 0.4, 0.01)
        st.caption('참고: 이전 PINN 최적점은 t = 0.287, m = 0.028 (p = 0.40)')
        up, lo = naca4(m, p, t)
        xu, yu, xl, yl = up[:, 0], up[:, 1], lo[:, 0], lo[:, 1]
        fe = naca_features(t, m, p)
        label = f'NACA형 t={t:.3f}, m={m:.3f}, p={p:.2f}'
        sel_file = None
    else:
        opts = feat.sort_values('file')
        names = [f"{r.file.replace('.dat', '')}  ({r['name']})" for _, r in opts.iterrows()]
        default = names.index(next(n for n in names if n.startswith('naca4412'))) if any(n.startswith('naca4412') for n in names) else 0
        choice = st.selectbox('익형', names, index=default)
        row = opts.iloc[names.index(choice)]
        sel_file = row.file
        s = shapes[sel_file]
        xu = xl = XS; yu, yl = s[:len(XS)], s[len(XS):]
        fe = {k: row[k] for k in ['t_max', 'x_t', 'camber_max', 'x_c', 'I']}
        label = sel_file.replace('.dat', '')
    re_label = st.radio('레이놀즈수 Re', list(RE_OPTIONS), index=1, horizontal=True)
    RE = RE_OPTIONS[re_label]
    st.divider()
    st.caption(f'대리모델 평균 오차(격자 325개, XFOIL 기준, Re 2×10⁵): 약 {err_all:.1f}%')

# 공통 계산
cl, cd = predict(models, fe, RE, ALPHA_CURVE)
ld = cl / cd
mask = (ALPHA_CURVE >= 0) & (ALPHA_CURVE <= 12)
i_best = np.argmax(np.where(mask, ld, -np.inf))
real_tm = feat[['t_max', 'camber_max']].values
nearby = int(np.sum(np.hypot(real_tm[:, 0] - fe['t_max'], real_tm[:, 1] - fe['camber_max']) < 0.02))

tab1, tab2, tab3, tab4 = st.tabs(['① 형상 분석', '② 강성–효율 지도', '③ 역설계', '탐구 요약'])

# ---------------- ① 형상 분석 ----------------
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xu, y=yu, mode='lines', name='윗면', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=xl, y=yl, mode='lines', name='아랫면', line=dict(color='#1f77b4')))
        pu, pl = naca4(PINN['m'], PINN['p'], PINN['t'])
        fig.add_trace(go.Scatter(x=np.r_[pu[:, 0], np.nan, pl[:, 0]], y=np.r_[pu[:, 1], np.nan, pl[:, 1]],
                                 mode='lines', name='이전 PINN 최적점', line=dict(color='red', dash='dot')))
        fig.update_layout(title=f'단면 형상: {label}', height=330, showlegend=True,
                          yaxis=dict(scaleanchor='x', scaleratio=1), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        a, b = st.columns(2)
        a.metric('최대 두께비', f"{fe['t_max']:.3f}")
        b.metric('최대 캠버', f"{fe['camber_max']:.3f}")
        a.metric('단면 이차모멘트 I (×10⁻⁵)', f"{fe['I']*1e5:.1f}")
        b.metric('I (PINN 최적점 대비)', f"{fe['I']/I_pinn*100:.0f}%")
        a.metric('최대 L/D (대리모델)', f'{ld[i_best]:.1f}')
        b.metric('최적 받음각', f'{ALPHA_CURVE[i_best]:.1f}°')

    # XFOIL 값이 있으면 같이 보여줌
    xf = lab[(lab.file == sel_file) & (lab.Re == RE)] if sel_file else None
    g1, g2 = st.columns(2)
    for col, y, yname in ((g1, cl, '양력계수 CL'), (g2, ld, '양항비 L/D')):
        f = go.Figure()
        f.add_trace(go.Scatter(x=ALPHA_CURVE, y=y, mode='lines', name='대리모델 예측'))
        if xf is not None and len(xf):
            f.add_trace(go.Scatter(x=xf.alpha, y=xf.CL if yname.startswith('양력') else xf.CL / xf.CD,
                                   mode='markers', name='XFOIL 계산값(학습 데이터)'))
        f.update_layout(title=f'{yname} – 받음각 (Re {re_label})', xaxis_title='받음각 α (°)', yaxis_title=yname,
                        height=320, margin=dict(l=10, r=10, t=40, b=10))
        col.plotly_chart(f, width='stretch')

    st.caption('곡선이 계단 모양인 것은 랜덤포레스트가 학습 데이터 구간별 평균값으로 예측하기 때문입니다.')

    # 신뢰도 안내 (탐구 결과에 근거)
    st.subheader('이 예측을 얼마나 믿을 수 있나')
    msgs = [f'주변(두께·캠버 차이 0.02 이내)의 학습용 실제 익형: **{nearby}개**']
    if fe['t_max'] >= 0.25:
        how = '**순위를 거의 맞히지 못했습니다**' if rho_thick < 0.6 else '순위를 어느 정도 맞혔습니다'
        msgs.append(f'⚠️ 두께 0.25 이상 영역: 이 탐구에서 대리모델은 이 영역 형상들의 {how} '
                    f'(PINN 최적점보다 강성이 큰 형상들의 순위 상관 {rho_thick:.2f}).')
    if RE == 100_000 and fe['t_max'] >= 0.25:
        msgs.append('⚠️ Re 1×10⁵ + 두꺼운 익형: XFOIL 자체가 이 조건을 제대로 풀지 못했습니다 (탐구 문제 3). 학습 데이터도 부정확할 수 있습니다.')
    if mode.startswith('NACA') and abs(p - 0.4) < 1e-9 and RE == 200_000:
        hit = grid[np.isclose(grid.t, round(t, 2)) & np.isclose(grid.m, round(m, 3))]
        if len(hit) and np.isclose(t, round(t, 2)) and np.isclose(m, round(m, 3)) and pd.notna(hit.iloc[0].XFOIL_200k):
            msgs.append(f'✅ 이 형상은 XFOIL로 직접 계산한 격자에 있습니다: XFOIL 최대 L/D = **{hit.iloc[0].XFOIL_200k:.1f}** '
                        f'(대리모델 {hit.iloc[0].LDmax_200k:.1f})')
    if xf is not None and len(xf):
        msgs.append('✅ 학습 데이터에 이 익형의 XFOIL 계산값이 있어 그래프에 점으로 표시했습니다.')
    for s_ in msgs:
        st.markdown('- ' + s_)

    if mode.startswith('NACA'):
        st.download_button('이 형상의 좌표 파일(.dat) 내려받기', dat_text(t, m, p, label),
                           file_name=f'naca_t{t:.3f}_m{m:.3f}_p{p:.2f}.dat')

# ---------------- ② 강성–효율 지도 ----------------
with tab2:
    st.markdown('NACA 4자리 형상 325개(캠버 위치 0.4)를 **XFOIL로 직접 계산**한 결과입니다 (Re 2×10⁵). '
                '파란 선은 강성과 효율을 동시에 더 좋게 만들 수 없는 형상들(파레토 전선)입니다.')
    g = gv.copy()
    front = pareto(g, 'I', 'XFOIL_200k')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.I * 1e5, y=g.XFOIL_200k, mode='markers', name='격자 형상 (XFOIL)',
                             marker=dict(color=g.t, colorscale='Viridis', size=7, colorbar=dict(title='t/c')),
                             text=[f't={a:.2f}, m={b:.3f}' for a, b in zip(g.t, g.m)],
                             hovertemplate='%{text}<br>I=%{x:.1f}e-5<br>L/D=%{y:.1f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=front.I * 1e5, y=front.XFOIL_200k, mode='lines', name='파레토 전선 (XFOIL)',
                             line=dict(color='#00b4d8', width=3)))
    pr = grid.iloc[-1]
    fig.add_trace(go.Scatter(x=[pr.I * 1e5], y=[pr.XFOIL_200k], mode='markers', name='이전 PINN 최적점 (XFOIL)',
                             marker=dict(symbol='star', size=18, color='red', line=dict(color='black', width=1))))
    if RE == 200_000:
        fig.add_trace(go.Scatter(x=[fe['I'] * 1e5], y=[ld[i_best]], mode='markers', name='현재 형상 (대리모델 예측)',
                                 marker=dict(symbol='diamond', size=14, color='orange', line=dict(color='black', width=1))))
    fig.update_layout(xaxis_title='단면 이차모멘트 I (×10⁻⁵, 시위 = 1)', yaxis_title='최대 L/D', height=560,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='left', x=0))
    st.plotly_chart(fig, width='stretch')
    if RE != 200_000:
        st.caption('현재 형상은 Re 2×10⁵를 선택했을 때만 지도에 표시됩니다 (지도는 Re 2×10⁵ 계산값).')

# ---------------- ③ 역설계 ----------------
with tab3:
    st.markdown('조건(최소 강성)을 정하면 그 조건을 만족하면서 최대 L/D가 큰 형상을 찾습니다 (Re 2×10⁵).')
    I_min = st.slider('최소 단면 이차모멘트 I (×10⁻⁵)', 1.0, 110.0, 20.0, 1.0) * 1e-5

    st.subheader('A. XFOIL로 검증된 답')
    pool = [gv.assign(p=0.4, LD=gv.XFOIL_200k, 출처='격자 탐색 (p = 0.4)')[['t', 'm', 'p', 'I', 'LD', '출처']]]
    if q4 is not None:
        q = q4.dropna(subset=['LD_xfoil'])
        pool.append(q.assign(LD=q.LD_xfoil, 출처='역설계 후보 (XFOIL 재계산)')[['t', 'm', 'p', 'I', 'LD', '출처']])
    pool = pd.concat(pool, ignore_index=True)
    gg = pool[pool.I >= I_min]
    st.caption(f'XFOIL로 직접 계산한 형상 {len(pool)}개(격자 {len(gv)}개 + 역설계 후보) 중에서 고릅니다.')
    if len(gg):
        b = gg.loc[gg.LD.idxmax()]
        st.success(f'XFOIL 최대 L/D 1위: t/c {b.t:.3f}, 캠버 {b.m:.4f}, 캠버 위치 {b.p:.2f} → '
                   f'L/D **{b.LD:.1f}**, I {b.I*1e5:.1f}×10⁻⁵  ({b["출처"]})')
        st.download_button('이 형상 좌표(.dat)', dat_text(b.t, b.m, b.p, f'XFOIL-verified best I>={I_min*1e5:.0f}e-5'),
                           file_name=f'verified_best_I{I_min*1e5:.0f}e-5.dat')
    else:
        st.warning('이 조건을 만족하는 검증된 형상이 없습니다.')
    if final is not None:
        st.markdown('탐구에서 역설계(유전 알고리즘) 상위 10개를 XFOIL로 다시 계산해 고른 최종 형상:')
        show = final.rename(columns={'I_x1e5': 'I(×10⁻⁵)', 'XFOIL_LD': 'XFOIL L/D', 'GA_LD': '역설계 최고', '격자_LD': '격자 최고'})
        st.dataframe(show.round(3), width='stretch', hide_index=True)

    st.subheader('B. 대리모델로 새로 찾기 (검증 안 됨)')
    if q4 is not None:
        q4ok = q4.dropna(subset=['LD_xfoil'])
        n_sc = q4.scenario.nunique()
        n_hit = int((q4[q4.rank_sur == 1].rank_xfoil == 1).sum())
        st.warning(f'이 탐구에서 대리모델로 찾은 1등이 XFOIL로도 1등인 경우는 **{n_hit}/{n_sc}개 조건**뿐이었고, '
                   f'상위 후보는 평균 {q4ok["err_%"].mean():+.0f}% 높게 예측되었습니다. '
                   '아래 결과는 후보일 뿐이며 XFOIL로 확인해야 합니다.')
    else:
        st.warning('대리모델로 찾은 결과는 후보일 뿐이며 XFOIL로 확인해야 합니다.')
    if st.button('유전 알고리즘 실행 (약 10~30초)'):
        LO, HI = np.array([0.06, 0.0, 0.2]), np.array([0.30, 0.06, 0.6])
        rng = np.random.default_rng(0)
        pop = LO + rng.random((40, 3)) * (HI - LO)
        arc, prog = [], st.progress(0.0)
        for gen in range(30):
            I, ldm = surrogate_ldmax_batch(models, pop, 200_000)
            fit = np.where(I >= I_min, ldm, ldm - 1000 * (I_min - I) / I_min - 100)
            arc.append(np.c_[pop, I, ldm])
            elite = pop[np.argsort(fit)[-4:]]
            kids = []
            while len(kids) < len(pop) - 4:
                i1, i2 = rng.choice(len(pop), 3, replace=False), rng.choice(len(pop), 3, replace=False)
                a_, b_ = pop[i1[np.argmax(fit[i1])]], pop[i2[np.argmax(fit[i2])]]
                child = a_ + rng.uniform(-0.25, 1.25, 3) * (b_ - a_)
                child += rng.normal(0, 0.05, 3) * (HI - LO) * (rng.random(3) < 0.3)
                kids.append(np.clip(child, LO, HI))
            pop = np.vstack([elite, kids]); prog.progress((gen + 1) / 30)
        arc = pd.DataFrame(np.vstack(arc), columns=['t', 'm', 'p', 'I', 'LD_sur'])
        top = arc[arc.I >= I_min].round({'t': 3, 'm': 4, 'p': 2}).drop_duplicates(['t', 'm', 'p'])
        top = top.sort_values('LD_sur', ascending=False).head(5)
        top['I(×10⁻⁵)'] = top.I * 1e5
        st.dataframe(top[['t', 'm', 'p', 'I(×10⁻⁵)', 'LD_sur']].rename(columns={'LD_sur': '대리모델 L/D (검증 안 됨)'}).round(3),
                     width='stretch', hide_index=True)
        b = top.iloc[0]
        st.download_button('1위 후보 좌표(.dat) — XFOIL로 확인 필요', dat_text(b.t, b.m, b.p, 'surrogate GA candidate (unverified)'),
                           file_name='ga_candidate_unverified.dat')

# ---------------- 탐구 요약 ----------------
with tab4:
    st.markdown(f"""
**데이터**: UIUC 익형 좌표 1,665개 → 정제 후 1,641개 → XFOIL(Re 1·2·5×10⁵, 받음각 −5~15°) 계산 → 학습 데이터 {len(lab):,}행, 익형 {lab.file.nunique():,}개

**이 도구의 대리모델**: 랜덤포레스트 (입력: 최대 두께·위치, 최대 캠버·위치, 받음각, log Re → 출력: CL, log CD).
웹 도구에서는 메모리 때문에 노트북(나무 300개)보다 가벼운 설정(나무 80개, 잎 최소 8개, 표본 50%)으로 다시 학습했습니다. 그래서 노트북의 예측값과 조금 다를 수 있습니다.

**대리모델 신뢰도 (XFOIL 직접 계산 325개 기준)**: 최대 L/D 평균 절대오차 {err_all:.1f}%,
두꺼운 영역(I ≥ PINN 최적점) 순위 상관 {rho_thick:.2f}

**핵심 결론**
- 이전 PINN 최적점(t/c 0.287, 캠버 0.028)은 강성–효율 파레토 전선 위에 있지 않음 (XFOIL 기준, Re 2×10⁵·5×10⁵)
- 대리모델 위에서 최적화한 1등은 XFOIL로 1등이 아니었음 → 최적화 결과는 반드시 XFOIL로 재확인
""")
