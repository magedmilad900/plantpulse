import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="PlantPulse", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Playfair+Display:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:radial-gradient(ellipse at 80% 0%,#10201b 0%,#07090b 45%);color:#f2f5f3}
.block-container{padding-top:1.5rem;max-width:1700px}
[data-testid="stSidebar"]{background:#080b0d;border-right:1px solid #20272c}
[data-testid="stSidebar"] *{color:#e8eceb}
h1,h2,h3,h4{font-family:'Manrope',sans-serif!important;color:#f4f6f5;letter-spacing:-.045em}
.fm-eyebrow{font-size:.68rem;letter-spacing:.19em;text-transform:uppercase;font-weight:800;color:#55dca5}
.fm-title{font-size:2.3rem;line-height:1.08;font-weight:800;letter-spacing:-.065em;margin:.35rem 0;color:#f5f7f6}
.fm-sub{font-size:.9rem;color:#87939a;line-height:1.65}
.fm-panel{background:linear-gradient(145deg,#11171a,#090c0e);border:1px solid #222b30;border-radius:16px;padding:18px 20px}
.fm-label{font-size:.74rem;color:#89969d}.fm-number{font-size:2rem;font-weight:800;letter-spacing:-.07em}
.fm-note{font-size:.73rem;color:#7f8b91}
.fm-status{display:inline-block;padding:4px 9px;border-radius:999px;font-size:.68rem;font-weight:800}
.ok{color:#5de5ad;background:#103326}.watch{color:#f6c766;background:#392c12}.bad{color:#ff7d86;background:#3a171b}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#11171a,#0b0e10);border:1px solid #222b30;padding:15px;border-radius:14px}
div[data-testid="stMetricLabel"]{color:#89969d} div[data-testid="stMetricValue"]{color:#f4f7f5;font-weight:800}
.stButton button,.stDownloadButton button{border:1px solid #2b383d;border-radius:10px;background:#101619;color:#f1f5f3;font-weight:700}
.stButton button[kind="primary"]{background:#31d995;color:#06130d;border-color:#31d995}
div[data-testid="stDataFrame"]{border:1px solid #222b30;border-radius:12px;overflow:hidden}
hr{border-color:#20272c}
</style>""", unsafe_allow_html=True)

MACHINES=["CNC-01","PRESS-02","PUMP-03","MOTOR-04","CONVEYOR-05","COMPRESSOR-06"]
BASE={"CNC-01":(67,3.2,7.4),"PRESS-02":(73,4.6,9.1),"PUMP-03":(58,2.4,5.8),"MOTOR-04":(64,3,6.7),"CONVEYOR-05":(55,2.1,5.2),"COMPRESSOR-06":(78,5.2,10.2)}
@st.cache_data
def make_data():
    rng=np.random.default_rng(42); rows=[]; now=pd.Timestamp.now().floor("h")
    for m in MACHINES:
        t,v,p=BASE[m]; n=72; drift=np.linspace(0,rng.uniform(0,13),n)
        temp=t+rng.normal(0,2.2,n)+drift*.55; vib=v+rng.normal(0,.28,n)+drift*.045
        power=p+rng.normal(0,.35,n)+drift*.06
        eff=np.clip(96-drift*.55+rng.normal(0,1,n),65,99)
        if m=="COMPRESSOR-06": temp[-8:]+=np.linspace(3,15,8); vib[-8:]+=np.linspace(.5,2.6,8)
        if m=="PRESS-02": vib[-5:]+=np.linspace(.6,2,5)
        for i in range(n): rows.append({"timestamp":now-pd.Timedelta(hours=n-i-1),"machine":m,"temperature_c":round(float(temp[i]),1),"vibration_mm_s":round(float(vib[i]),2),"power_kw":round(float(power[i]),2),"efficiency":round(float(eff[i]),1)})
    return pd.DataFrame(rows)
def score(df,tl,vl,pl):
    x=df.copy(); te=np.clip((x.temperature_c-tl)/max(tl*.18,1),0,1); ve=np.clip((x.vibration_mm_s-vl)/max(vl*1.2,.1),0,1); pe=np.clip((x.power_kw-pl)/max(pl*.25,1),0,1)
    x["risk_score"]=(100*(.42*te+.38*ve+.20*pe)).round().astype(int).clip(0,100)
    x["status"]=np.select([x.risk_score>=65,x.risk_score>=30],["Critical","Watch"],default="Healthy")
    return x
def header(k,t,s): st.markdown(f'<div class="fm-eyebrow">{k}</div><div class="fm-title">{t}</div><div class="fm-sub">{s}</div>',unsafe_allow_html=True)
def pill(s): return f'<span class="fm-status {"bad" if s=="Critical" else "watch" if s=="Watch" else "ok"}">{s.upper()}</span>'
def theme(fig,h=320):
    fig.update_layout(height=h,margin=dict(l=4,r=8,t=22,b=4),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Manrope",color="#aab5b9"),legend_title_text="")
    fig.update_xaxes(gridcolor="#20282c",linecolor="#20282c",zeroline=False); fig.update_yaxes(gridcolor="#20282c",linecolor="#20282c",zeroline=False)
    return fig

st.sidebar.markdown('<div style="padding:12px 0 22px"><div style="font-size:1.8rem">〰</div><div style="font-weight:800;font-size:1.35rem;letter-spacing:-.06em">PlantPulse</div><div style="font-size:.72rem;color:#7f8c91">PREDICTIVE OPERATIONS FOR INDUSTRY</div></div>',unsafe_allow_html=True)
page=st.sidebar.radio("WORKSPACE",["Overview","Assets","Insights","Maintenance","Exports","Settings","Model notes"])
st.sidebar.markdown("---"); st.sidebar.markdown('<div class="fm-eyebrow">CONFIGURATION</div>',unsafe_allow_html=True)
tl=st.sidebar.slider("Temperature limit · °C",50,110,85); vl=st.sidebar.slider("Vibration limit · mm/s",1.0,10.0,6.0,.5); plim=st.sidebar.slider("Power limit · kW",4.0,18.0,11.0,.5)
st.sidebar.markdown("<div style='margin-top:22px;padding:12px;border:1px solid #24332e;border-radius:12px;background:#0d1715'><b style='color:#f2f5f3'>Maged Gad</b><br><span style='color:#7f8b91;font-size:.72rem'>Operations lead</span></div>", unsafe_allow_html=True)
st.sidebar.caption("Simulated sensor data · demo only")
all_data=score(make_data(),tl,vl,plim); latest=all_data.sort_values("timestamp").groupby("machine").tail(1).sort_values("risk_score",ascending=False)
critical=int((latest.status=="Critical").sum()); watch=int((latest.status=="Watch").sum()); healthy=int((latest.status=="Healthy").sum()); avg=int(latest.risk_score.mean()); top=latest.iloc[0]
if page=="Overview":
    header("INDUSTRIAL INTELLIGENCE / OVERVIEW","Clarity for every machine.","An executive view of machine health, operational risk, and the next best maintenance action.")
    st.markdown(f'<div class="fm-panel" style="display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap"><div><div class="fm-eyebrow">FACTORY STATUS</div><div style="font-size:1.2rem;font-weight:800;margin-top:5px">All systems monitored <span style="color:#31d995">●</span></div><div class="fm-note">Simulated update · {datetime.now().strftime("%d %b %Y, %H:%M")}</div></div><div><div class="fm-note">TOP PRIORITY</div><div style="font-size:1.4rem;font-weight:800">{top.machine} &nbsp; {pill(top.status)}</div><div class="fm-note">Risk indicator {top.risk_score}/100</div></div></div>',unsafe_allow_html=True)
    st.write(""); a,b,c,d=st.columns(4); a.metric("Total machines",6,"Fleet monitored"); b.metric("Healthy",healthy,f"{healthy/6:.0%} of fleet"); c.metric("Needs attention",watch,"Watch list"); d.metric("Critical",critical,"Review first")
    st.write(""); x,y,z=st.columns([1.35,1,1])
    with x:
        st.markdown("### Factory overview"); st.markdown('<div class="fm-sub">Latest risk indicator by machine</div>',unsafe_allow_html=True)
        f=px.bar(latest.sort_values("risk_score"),x="risk_score",y="machine",orientation="h",color="status",color_discrete_map={"Healthy":"#31d995","Watch":"#f3bd55","Critical":"#ff5d68"},labels={"risk_score":"Risk indicator","machine":"","status":"Status"}); f.update_layout(bargap=.4); f.update_xaxes(range=[0,100]); st.plotly_chart(theme(f,340),use_container_width=True)
    with y:
        st.markdown("### Risk distribution"); counts=latest.status.value_counts().reindex(["Healthy","Watch","Critical"],fill_value=0).reset_index(); counts.columns=["Status","Machines"]
        f=px.pie(counts,names="Status",values="Machines",hole=.72,color="Status",color_discrete_map={"Healthy":"#31d995","Watch":"#f3bd55","Critical":"#ff5d68"}); f.update_traces(textinfo="none"); f.add_annotation(text=f"<b>{avg}</b><br><span style='font-size:10px'>AVG RISK</span>",showarrow=False,font=dict(color="#f1f5f3",size=21)); st.plotly_chart(theme(f,340),use_container_width=True)
    with z:
        st.markdown("### Recent alerts"); flagged=latest[latest.status!="Healthy"]
        if flagged.empty: st.success("No alerts at current thresholds.")
        for _,r in flagged.iterrows(): st.markdown(f'<div class="fm-panel" style="padding:12px;margin-bottom:9px"><div style="display:flex;justify-content:space-between"><b>{r.machine}</b>{pill(r.status)}</div><div class="fm-note">Temp {r.temperature_c}°C · Vibration {r.vibration_mm_s} mm/s</div><div class="fm-note">Risk {r.risk_score}/100</div></div>',unsafe_allow_html=True)
    st.write(""); left,right=st.columns([1.2,1])
    with left:
        st.markdown("### Operational efficiency"); tr=all_data.groupby("timestamp",as_index=False).efficiency.mean(); f=px.area(tr,x="timestamp",y="efficiency"); f.update_traces(line_color="#31d995",fillcolor="rgba(49,217,149,.10)"); f.update_yaxes(title="Efficiency · demo index",range=[60,100]); st.plotly_chart(theme(f,260),use_container_width=True)
    with right:
        st.markdown("### Latest machine readings"); st.dataframe(latest[["machine","temperature_c","vibration_mm_s","power_kw","risk_score","status"]].rename(columns={"machine":"Machine","temperature_c":"Temp °C","vibration_mm_s":"Vibration","power_kw":"Power kW","risk_score":"Risk","status":"Status"}),use_container_width=True,hide_index=True,height=260)
elif page=="Assets":
    header("ASSET MANAGEMENT","Machine intelligence.","Explore individual equipment, review signal history, and understand what changed.")
    m=st.selectbox("Choose a machine",MACHINES,index=MACHINES.index(top.machine)); hist=all_data[all_data.machine==m].sort_values("timestamp"); r=latest[latest.machine==m].iloc[0]
    a,b,c,d=st.columns(4); a.metric("Temperature",f"{r.temperature_c} °C"); b.metric("Vibration",f"{r.vibration_mm_s} mm/s"); c.metric("Power",f"{r.power_kw} kW"); d.metric("Risk",f"{r.risk_score}/100",r.status)
    sig=st.radio("Sensor signal",["Temperature","Vibration","Power"],horizontal=True); col,label,lim={"Temperature":("temperature_c","Temperature · °C",tl),"Vibration":("vibration_mm_s","Vibration · mm/s",vl),"Power":("power_kw","Power · kW",plim)}[sig]
    f=go.Figure(); f.add_trace(go.Scatter(x=hist.timestamp,y=hist[col],mode="lines",line=dict(color="#31d995",width=2.5),fill="tozeroy",fillcolor="rgba(49,217,149,.08)",name=label)); f.add_hline(y=lim,line_dash="dash",line_color="#ff5d68",annotation_text="Configured threshold"); f.update_layout(yaxis_title=label); st.plotly_chart(theme(f,390),use_container_width=True); st.markdown("### Recent signal records"); st.dataframe(hist.tail(24),use_container_width=True,hide_index=True)
elif page=="Insights":
    header("PERFORMANCE & TRENDS","See the signals behind the score.","Compare machines and explore trends against configured thresholds.")
    metric=st.selectbox("Compare metric",["temperature_c","vibration_mm_s","power_kw","risk_score"],format_func=lambda v:{"temperature_c":"Temperature · °C","vibration_mm_s":"Vibration · mm/s","power_kw":"Power · kW","risk_score":"Risk indicator"}[v])
    f=px.line(all_data,x="timestamp",y=metric,color="machine",color_discrete_sequence=["#31d995","#f3bd55","#ff5d68","#73a9ff","#c49aff","#ff9b71"]); st.plotly_chart(theme(f,440),use_container_width=True)
    st.markdown("### Correlation snapshot"); f=px.imshow(all_data[["temperature_c","vibration_mm_s","power_kw","risk_score","efficiency"]].corr(),text_auto=".2f",color_continuous_scale=["#101619","#174b3b","#31d995"],zmin=-1,zmax=1,aspect="auto"); st.plotly_chart(theme(f,390),use_container_width=True); st.caption("Correlations use simulated demo data, not real industrial findings.")
elif page=="Maintenance":
    header("MAINTENANCE OPERATIONS","Alerts & recommended checks.","A transparent triage queue showing which configured thresholds were crossed.")
    alerts=latest[latest.status!="Healthy"]
    if alerts.empty: st.success("No machines above configured thresholds. Adjust limits in the sidebar to test.")
    for _,r in alerts.iterrows():
        with st.container(border=True):
            a,b=st.columns([4,1])
            with a:
                st.markdown(f"### {r.machine}  {pill(r.status)}",unsafe_allow_html=True); reasons=[]
                if r.temperature_c>tl: reasons.append(f"Temperature {r.temperature_c}°C exceeds {tl}°C")
                if r.vibration_mm_s>vl: reasons.append(f"Vibration {r.vibration_mm_s} mm/s exceeds {vl} mm/s")
                if r.power_kw>plim: reasons.append(f"Power {r.power_kw} kW exceeds {plim} kW")
                st.write("**Detected signals:** "+"; ".join(reasons or ["Combined signals nearing thresholds."]))
                st.write("**Suggested checks:** verify sensor integrity, inspect cooling and lubrication, check bearings/alignment, and compare operating load with baseline.")
            with b: st.metric("Risk",f"{r.risk_score}/100")
    st.download_button("Export alert queue as CSV",alerts.to_csv(index=False).encode(),file_name="plantpulse_alerts.csv",mime="text/csv")
elif page=="Exports":
    header("REPORTING","Export your assessment.","Download the current simulated snapshot for review or presentations.")
    st.markdown('<div class="fm-panel"><div class="fm-eyebrow">CURRENT SNAPSHOT</div><div class="fm-number">6 assets</div><div class="fm-sub">Readings, risk indicators, statuses and timestamps.</div></div>',unsafe_allow_html=True); st.write("")
    st.download_button("Download full sensor history · CSV",all_data.to_csv(index=False).encode(),file_name="plantpulse_sensor_history.csv",mime="text/csv",type="primary")
    st.download_button("Download latest summary · CSV",latest.to_csv(index=False).encode(),file_name="plantpulse_summary.csv",mime="text/csv")
    st.write(f"Current snapshot: **{healthy} healthy**, **{watch} watch**, **{critical} critical**. Average risk indicator: **{avg}/100**.")
elif page=="Settings":
    header("WORKSPACE PREFERENCES","Make it yours.","Settings for this demo workspace.")
    st.markdown('<div class="fm-panel"><b>PlantPulse</b><p class="fm-sub">Predictive Care for Industry</p><b>Data source</b><p class="fm-sub">Synthetic sensor readings generated locally.</p><b>Alert bands</b><p class="fm-sub">Healthy: below 30 · Watch: 30–64 · Critical: 65 and above.</p><b>Visual system</b><p class="fm-sub">Obsidian black · emerald signal · editorial typography.</p></div>',unsafe_allow_html=True); st.info("Thresholds are adjustable from the sidebar.")
else:
    header("MODEL CARD","Transparent by design.","A hackathon proof of concept demonstrating an explainable workflow.")
    st.markdown("""<div class="fm-panel"><h3>How it works</h3><p class="fm-sub">Synthetic temperature, vibration and power readings are compared with configurable thresholds. Normalized exceedances combine into a 0–100 triage indicator.</p><h3>Limitations</h3><ul><li>This is not a trained AI failure predictor.</li><li>The score is not a calibrated probability of failure.</li><li>Synthetic data cannot establish real-world accuracy or impact.</li><li>Never use this demo to control equipment or replace qualified technicians.</li></ul><h3>Next validation steps</h3><ol><li>Use a documented public predictive-maintenance dataset.</li><li>Define the target and avoid data leakage.</li><li>Compare threshold rules against a trained model.</li><li>Report precision, recall, false alarms and warning lead time.</li></ol></div>""",unsafe_allow_html=True)
st.markdown('<div style="margin-top:28px;border-top:1px solid #20272c;padding-top:15px;display:flex;justify-content:space-between;color:#69767d;font-size:.72rem"><span>◈ PlantPulse · v2.0</span><span>Predict · Prevent · Keep production running</span></div>',unsafe_allow_html=True)
