import os
import datetime
import json
import urllib.request
import urllib.parse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

base_url = os.getenv("FORECAST_API", "http://127.0.0.1:8000/v1/forecast")

class Forecast:

    def __init__(self, base_url):
        self.base_url = base_url

    @st.cache
    def get_data(self, geo_id: str, from_date: datetime.datetime, to_date: datetime.datetime):
        dtf = datetime.datetime.combine(from_date, datetime.datetime.min.time())
        dtt = datetime.datetime.combine(to_date, datetime.datetime.min.time())
        geo_id_encode = urllib.parse.quote(geo_id)
        url = f"{self.base_url}/getMetric"
        interval = {"geo_id": geo_id, "metric": "cumulative_confirmed", "method": "curvefit", "time_from": dtf.isoformat(), "time_to": dtt.isoformat()}
        interval_payload = json.dumps(interval).encode("utf-8")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, data=interval_payload)
        r = urllib.request.urlopen(req).read()
        data = json.loads(r.decode("utf-8"))
        df = pd.DataFrame(data["result"]["series"]).set_index("date_id")
        return df

    @st.cache
    def get_geo_id(self):
        url = f"{self.base_url}/geo_id"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        r = urllib.request.urlopen(req).read()
        data = json.loads(r.decode("utf-8"))
        return sorted(data)

    def write(self):
        with st.spinner("Loading Forecast ..."):
            geos = self.get_geo_id()
            geo = st.selectbox("Select Geo Id:", geos)
            from_date = st.date_input("From", datetime.date(2020, 1, 1))
            to_date = st.date_input(
                "To", datetime.datetime.now() + datetime.timedelta(days=14)
            )

            df = self.get_data(geo, from_date, to_date)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["forecast"],
                    mode="lines+markers",
                    name="Forecast",
                    line_color="red",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["actual"],
                    mode="lines+markers",
                    name="cumulative cases",
                    line_color="rgba(132,183,83,1.0)",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["credible_interval_low"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,0,0,0.0)",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["credible_interval_high"],
                    fill="tonexty",  # fill area between this and previous trace
                    mode="lines",
                    line_color="rgba(0,0,0,0.0)",
                    fillcolor="rgba(0,0,0,0.1)",
                    name="95% credible interval",
                )
            )

            st.plotly_chart(fig)

MENU = {
  "Forecast": Forecast(base_url)
}

def main():

    st.sidebar.title("Menu")
    selection = st.sidebar.selectbox("Go to", list(MENU.keys()))
    menu = MENU[selection]

    with st.spinner(f"Loading {selection} ..."):
        menu.write()

if __name__ == "__main__":
    main()
