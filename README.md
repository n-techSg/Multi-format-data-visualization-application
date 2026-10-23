# Plotbench for Streamlit

Plot columns from CSV and Excel files, add calculated series, and ask Claude to
explain a chart or suggest new ones.

## Features

- Load CSV, TSV, XLS and XLSX files (choose the sheet, toggle the header row).
- 13 chart types: line, area, step, scatter, bubble, vertical/horizontal/stacked bar,
  histogram, pie, doughnut, polar area, radar.
- Pick any column for X and one or more for Y. Edit the chart title and axis labels.
- Combine rows sharing an X value (sum, average, count, min, max), log scale, sorting, zoom direction.
- Calculated series: quick A − B subtraction, or custom formulas such as
  `[ME fuel (t/day)] / [Speed (kn)]`, `[Fuel] - prev[Fuel]`, `cumsum[Fuel]`, `[Fuel] / mean[Fuel]`.
  Calculated series are drawn dashed and can use a right-hand axis.
- Choose which series appear in the chart and the saved PNG.
- Zoom by dragging, double-click to reset, save a PNG from the chart toolbar (camera icon).
- "Focus on an X range" limits both the chart and the AI explanation to part of the data.
- Optional AI: **Explain** a chart (with follow-up questions) and **Suggest charts** for a file.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Turn on the AI features

1. Create an API key at https://console.anthropic.com. API usage is billed to that account,
   separately from a Claude.ai subscription.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste in the key.
3. Restart the app. Without a key, everything except the AI buttons still works.

The app sends Claude a summary of the chart (statistics, correlations and up to about
150 sampled points). For suggestions it sends the column list with simple statistics and
the first 8 rows. Whole files are not sent.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository. The `.gitignore` keeps `secrets.toml` out.
2. At https://share.streamlit.io choose **Create app**, then pick the repo and `app.py`.
3. In **Advanced settings > Secrets**, paste the contents of your `secrets.toml`.
4. Deploy.

Community Cloud apps are public by default. For client or fleet data, restrict viewers in
the app's sharing settings, or run it on a private server instead.
