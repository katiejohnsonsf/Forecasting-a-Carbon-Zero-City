import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd

df_ts = pd.read_csv('stationary-data/all_df.csv')
df_ro = pd.read_csv('stationary-data/preds_min_df.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='Forecasting a Carbon Zero City'),
    html.P(children='A project by @astateofkate'),
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='phase-slider',
        min=df_ts['phase'].min(),
        max=df_ts['phase'].max(),
        value=df_ts['phase'].min(),
        marks={str(phase): str(phase) for phase in df_ts['phase'].unique()},
        step=None
    )
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('phase-slider', 'value'))
def update_figure(selected_phase):
    filtered_df = df_ts[df_ts.phase == selected_phase]

    fig = px.line(filtered_df, x="Date", y="kwh avg",
                  hover_name="kwh avg")

    fig.update_layout(transition_duration=500, 
                      title='Energy Efficiency Improvements Jan 2020-Jan 2022')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)