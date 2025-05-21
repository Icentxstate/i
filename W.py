# 📍 Interactive Map Tab (Streamlit + Folium)
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from branca.colormap import linear
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# ---- Layout setup ----
st.set_page_config(page_title="Hydro Dashboard", layout="wide")
st.title("🌍 Interactive Water Quality Map")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("INPUT_1.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])
    return df

df = load_data()

# ---- Pick Parameter for Coloring ----
numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_param = st.selectbox("🎯 Select parameter to color markers", numeric_cols)

# ---- Group by site for mapping ----
site_stats = df.groupby(['Site ID', 'Site Name', 'Latitude', 'Longitude'])[selected_param].mean().reset_index()

# ---- Map Base ----
m = folium.Map(location=[site_stats['Latitude'].mean(), site_stats['Longitude'].mean()], zoom_start=8, tiles="CartoDB positron")

# ---- Color Map ----
colormap = linear.YlGnBu_09.scale(site_stats[selected_param].min(), site_stats[selected_param].max())
colormap.caption = f"Mean {selected_param}"
colormap.add_to(m)

# ---- MarkerCluster + Points ----
marker_cluster = MarkerCluster().add_to(m)

for _, row in site_stats.iterrows():
    popup_content = f"""
    <b>{row['Site Name']}</b><br>
    Site ID: {row['Site ID']}<br>
    Avg {selected_param}: {row[selected_param]:.2f}
    """
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=8,
        color=colormap(row[selected_param]),
        fill=True,
        fill_opacity=0.8,
        popup=popup_content,
        tooltip=row['Site Name']
    ).add_to(marker_cluster)

# ---- Show Map ----
st_data = st_folium(m, width=1200, height=600)
# ----- تابع برای ساخت نمودار کوچک برای هر ایستگاه -----
def make_mini_chart(site_df, param='E. coli (MPN/100 mL)'):
    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.plot(site_df['Date'], site_df[param], color='teal', linewidth=1)
    ax.set_title('')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{encoded}"/>'

# --- اضافه کردن Marker ها با نمودار ---
for _, row in site_stats.iterrows():
    site_id = row['Site ID']
    site_name = row['Site Name']
    site_df = df[df['Site ID'] == site_id].sort_values('Date')

    if len(site_df) < 3:
        continue  # Ignore too small sites

    chart_html = make_mini_chart(site_df, selected_param)

    popup_content = f"""
    <div style="width:200px">
        <b>{site_name}</b><br>
        Site ID: {site_id}<br>
        Avg {selected_param}: {row[selected_param]:.2f}<br>
        {chart_html}<br>
        <a href="#graph" target="_self">🔍 View full analysis</a>
    </div>
    """

    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=8,
        color=colormap(row[selected_param]),
        fill=True,
        fill_opacity=0.9,
        popup=folium.Popup(popup_content, max_width=300),
        tooltip=site_name
    ).add_to(marker_cluster)

st.header("📈 Advanced Graph Analysis")

# ----- پارامترهای کاربر -----
selected_site_label = st.selectbox("🏞️ Select Site", site_options['label'])
selected_site_id = site_dict[selected_site_label]
selected_param = st.selectbox("📊 Select Parameter", numeric_cols)

# فیلتر تاریخ
min_date = pd.to_datetime(df['Date'].min())
max_date = pd.to_datetime(df['Date'].max())
date_range = st.slider("📅 Filter by Date Range", min_value=min_date, max_value=max_date,
                       value=(min_date, max_date))

# ----- استخراج داده -----
site_df = df[df['Site ID'] == selected_site_id].sort_values('Date')
site_df = site_df[(site_df['Date'] >= date_range[0]) & (site_df['Date'] <= date_range[1])]

# ----- میانگین متحرک -----
window_size = st.slider("📏 Moving Average Window (Days)", 2, 30, 7)
site_df['MA'] = site_df[selected_param].rolling(window=window_size).mean()

# ----- مقدار هشدار -----
custom_thresh = st.number_input(f"🚨 Optional Threshold for {selected_param}", value=0.0)

# ----- رسم نمودار -----
fig = go.Figure()

# مقادیر اصلی
fig.add_trace(go.Scatter(x=site_df['Date'], y=site_df[selected_param],
                         mode='lines+markers', name=selected_param,
                         line=dict(color='seagreen')))

# میانگین متحرک
fig.add_trace(go.Scatter(x=site_df['Date'], y=site_df['MA'],
                         mode='lines', name=f"{window_size}-day MA",
                         line=dict(color='orange', dash='dot')))

# خط هشدار
if custom_thresh > 0:
    fig.add_hline(y=custom_thresh, line=dict(color='red', dash='dash'),
                  annotation_text="Threshold", annotation_position="top left")

# تنظیمات گراف
fig.update_layout(
    title=f"{selected_param} Trend at {selected_site_label}",
    xaxis_title="Date", yaxis_title=selected_param,
    height=500, margin=dict(l=40, r=40, t=50, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# خلاصه آماری
st.markdown("### 📋 Summary Statistics")
st.write(site_df[selected_param].describe().round(2))

# -------------------- 🧭 Alert View Tab --------------------
import streamlit as st
import pandas as pd

st.header("🧭 Alert View – High Risk Events")

# ------ پارامتر، سایت و آستانه ------
selected_param = st.selectbox("📊 Select Parameter to Monitor", numeric_cols)
threshold = st.number_input(f"🚨 Alert Threshold for {selected_param}", value=100.0)

site_options_alert = ['All'] + sorted(df['Site Name'].unique().tolist())
selected_site_name = st.selectbox("🏞️ Select Site (or All)", site_options_alert)

# ------ فیلتر تاریخ ------
min_date_alert = pd.to_datetime(df['Date'].min())
max_date_alert = pd.to_datetime(df['Date'].max())
date_range_alert = st.slider("📅 Filter by Date Range", min_value=min_date_alert, max_value=max_date_alert,
                             value=(min_date_alert, max_date_alert))

# ------ فیلتر داده ------
df_alert = df.copy()
df_alert['Date'] = pd.to_datetime(df_alert['Date'])
df_alert = df_alert[(df_alert['Date'] >= date_range_alert[0]) & (df_alert['Date'] <= date_range_alert[1])]

if selected_site_name != "All":
    df_alert = df_alert[df_alert['Site Name'] == selected_site_name]

# ------ استخراج هشدارها ------
alerts_df = df_alert[df_alert[selected_param] > threshold]

st.markdown(f"### ⚠️ Found {len(alerts_df)} Alert(s) Above Threshold")

if not alerts_df.empty:
    st.dataframe(alerts_df[['Date', 'Site Name', 'Site ID', selected_param, 'Latitude', 'Longitude']].sort_values("Date"))

    # Optional: download alerts
    csv = alerts_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Alerts as CSV", csv, f"alerts_{selected_param}.csv", "text/csv")
else:
    st.info("✅ No alerts found in the selected range.")
# -------------------- 📤 Export Raw Data Tab --------------------
import streamlit as st

st.header("📤 Export & View Raw Data")

# ایستگاه و پارامتر
export_site_options = ['All'] + sorted(df['Site Name'].unique().tolist())
selected_site_export = st.selectbox("🏞️ Select Site (or All)", export_site_options)

selected_params_export = st.multiselect("📊 Select Parameter(s) to Export", numeric_cols, default=numeric_cols[:3])

# تاریخ
date_range_export = st.slider("📅 Select Date Range", min_value=min_date, max_value=max_date,
                              value=(min_date, max_date))

# فیلتر داده‌ها
df_export = df.copy()
df_export['Date'] = pd.to_datetime(df_export['Date'])
df_export = df_export[(df_export['Date'] >= date_range_export[0]) & (df_export['Date'] <= date_range_export[1])]

if selected_site_export != "All":
    df_export = df_export[df_export['Site Name'] == selected_site_export]

# فقط ستون‌های انتخابی
columns_to_show = ['Date', 'Site ID', 'Site Name', 'Latitude', 'Longitude'] + selected_params_export
df_show = df_export[columns_to_show].sort_values("Date")

st.markdown("### 📋 Filtered Data Preview")
st.dataframe(df_show, use_container_width=True)

# خروجی CSV
csv_export = df_show.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download CSV", csv_export, "filtered_data.csv", "text/csv")

# خروجی Excel
from io import BytesIO
excel_buffer = BytesIO()
df_show.to_excel(excel_buffer, index=False, engine='openpyxl')
st.download_button("⬇️ Download Excel", excel_buffer.getvalue(), "filtered_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
# -------------------- 🏁 Summary Dashboard Tab --------------------
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.title("🏁 Water Quality Summary Dashboard")

# لیست پارامترهای کلیدی برای نمایش کارت
summary_params = ['E. coli (MPN/100 mL)', 'DO (mg/L)', 'Temperature (°C)', 'Flow (CFS)']

# تاریخ فعلی و بازه ۶ ماه گذشته
latest_date = df['Date'].max()
six_months_ago = pd.to_datetime(latest_date) - pd.DateOffset(months=6)

df_recent = df[df['Date'] >= six_months_ago]

# طراحی کارت‌های آماری
cols = st.columns(len(summary_params))

for i, param in enumerate(summary_params):
    with cols[i]:
        # آمار
        mean_val = df_recent[param].mean()
        unit = param.split('(')[-1].replace(')', '')

        # رنگ بر اساس مقدار
        if "E. coli" in param:
            color = "#FF6666" if mean_val > 126 else "#66CC66"
            icon = "🦠"
        elif "DO" in param:
            color = "#FFDD57" if mean_val < 5 else "#66CC66"
            icon = "💨"
        elif "Temp" in param:
            color = "#87CEFA"
            icon = "🌡️"
        else:
            color = "#AAAAFF"
            icon = "💧"

        st.markdown(f"""
            <div style="background-color:{color}; padding:15px; border-radius:15px; text-align:center">
            <h3 style='margin-bottom:0'>{icon} {param.split('(')[0]}</h3>
            <h2 style='margin-top:5px'>{mean_val:.1f} {unit}</h2>
            </div>
        """, unsafe_allow_html=True)

# 🔄 روند Sparkline برای E. coli (نمونه)
st.markdown("### 📈 6-Month Trend for E. coli")
fig = go.Figure()
trend_df = df_recent.groupby('Date')['E. coli (MPN/100 mL)'].mean().reset_index()
fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['E. coli (MPN/100 mL)'],
                         mode='lines+markers', name='E. coli', line=dict(color='crimson')))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
# به جای colormap، رنگ با منطق هشدار:
def get_alert_color(val, param):
    if "E. coli" in param:
        return 'red' if val > 126 else 'green'
    elif "DO" in param:
        return 'red' if val < 5 else 'green'
    elif "Temperature" in param:
        return 'orange' if val > 30 else 'blue'
    else:
        return 'gray'

# و در Marker:
for _, row in site_stats.iterrows():
    color = get_alert_color(row[selected_param], selected_param)

    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=8,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(...),
        tooltip=row['Site Name']
    ).add_to(marker_cluster)
