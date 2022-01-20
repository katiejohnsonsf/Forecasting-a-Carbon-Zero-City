import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

df_ts = pd.read_csv('stationary-data/all_ci_df.csv')
df_ro = pd.read_csv('stationary-data/preds_min_df.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='Analytics for a Carbon Zero City'),
    dcc.Graph(id='graph-with-slider', style={'marginLeft': 10, 'marginRight': 10, 'marginTop': 10, 'marginBottom': 10, }),
    dcc.Slider(
        id='phase-slider',
        min=df_ts['phase'].min(),
        max=df_ts['phase'].max(),
        value=df_ts['phase'].min(),
        marks={
            0: {'label': 'Business as Usual'},
            1: {'label': '1,346 properties'},
            2: {'label': '2,692 properties'},
            3: {'label': '4,038 properties'},
            4: {'label': '5,384 properties'}
        },
        step=None
    )
], style={'marginLeft': '50px', 'marginTop': '50px', 'marginRight': '50px'})


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('phase-slider', 'value'))
def update_figure(selected_phase):
    filtered_df = df_ts[df_ts.phase == selected_phase]

    x = df_ts.Date
    y_upper = df_ts.upper_ci
    y_lower = df_ts.lower_ci

    fig = px.line(filtered_df, x="Date", y="kwh avg",
                  hover_name="kwh avg")

    fig.add_trace(go.Scatter(
        x=x+x[::-1], 
        y=y_upper+y_lower[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
    ))

    fig.update_layout(yaxis_range=[0,2200],
                      showlegend=False,
                      transition_duration=500, 
                      title='Impact of Energy Efficiency Improvements in 5,384 / 100,313 properties in Gainesville, FL')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)