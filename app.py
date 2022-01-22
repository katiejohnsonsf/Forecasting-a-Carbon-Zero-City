import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

df_ts = pd.read_csv('/home/katiejohnsonsf/mysite/stationary-data/all_ci_df.csv')
df_ro = pd.read_csv('/home/katiejohnsonsf/mysite/stationary-data/preds_min_df.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='Analytics for a Carbon Zero City'),
    dcc.Graph(id='graph-with-slider', style={'marginLeft': 10, 'marginRight': 10, 'marginTop': 10, 'marginBottom': 10, }),
    html.Div(dcc.Slider(
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
    ), style={'marginLeft': 10, 'marginRight': 10, 'marginTop': 10, 'marginBottom': 50, }),
    dcc.Markdown('''
#### Intended Audience

This dashboard helps people in the energy efficiency field prioritize energy efficiency projects for maximum impact and allows them to see how their work is impacting monthly average kWh per capita. It allows program managers at local non-profits, utilities, and energy efficiency retrofitting companies to size markets and prioritize reduction opportunities for a city. It helps answer the questions:
> 1. What is the optimal approach for addressing energy efficiency opportunities in a city?
> 2. To what extent will those energy efficiency improvements help a city get closer to its Carbon Zero goal?


#### Story

Energy efficiency improvements for grid electrical consumption is a core strategy in Gainesville’s Climate Action Plan for achieving carbon zero goal by 2035. This dashboard aligns with Gainesville’s long-term carbon zero goal by tracking how 5,384 energy efficiency improvements impact the city's KPI for grid-electrical emissions, *monthly average kWh per capita* (see y-axis above). The property reductions impacted in the two-year reporting cycle make up ~5% of all buildings in Gainesville, but their 100% improvement reduces grid electrical consumption by 34% between January 2020 and January 2023. Moving the slider from left to right shows the reductions impact of these projects as they are completed. Planning over a two-year period allows people to contribute more actively to Gainesville’s Climate Action Plan 2-year reporting cycle, which allows the city to track its overall progress.

Throughout each two-year reporting cycle, the data supports action and better decision-making by identifying the best energy efficiency opportunities in a city to tackle first. The energy efficiency projects are prioritized based on the difference between the predicted and actual ICC building code release used in a building’s construction. If a property is predicted to have used the 1990 building code but is recorded as having used the 1996 code release, there is a higher likelihood of more significant reductions. This prioritization method allows all properties in the dataset to be ranked from highest to lowest by reduction opportunity. This prioritization reduces more emissions for the city and has increased financial benefits for residents and energy efficiency companies.

Project by [Katie Johnson](https://www.linkedin.com/in/hellokatiejohnson/). More information on data sources and modeling methods are available on Github [here](https://github.com/katiejohnsonsf/Forecasting-a-Carbon-Zero-City).

''')
], style={'marginLeft': '75px', 'marginTop': '75px', 'marginRight': '75px', 'marginBottom': '75px'})


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