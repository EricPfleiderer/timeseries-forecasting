import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px

normalization = ['None', 'MinMax', 'Standard', 'Robust']
log_transform = ['No', 'Yes']

OPTIONS_STYLE = {'display': 'inline-block',
                 'padding-left': '20px',
                 'padding-right':  '20px'}


def build_dashboard(data):

    features = data[0:2]

    app = Dash(__name__)
    app.layout = html.Div([

        # Title
        html.Div(children='Energy Public Day Ahead Market (DAM), Canada, JUL-21 to JAN-23',
                 style={'textAlign': 'Center', 'color': 'blue', 'fontSize': 30}),
        html.Hr(),

        html.Div([dcc.RadioItems(options=features, value=features[0], id='displayed-feature-radio-item', inline=True)],
                 style={'text-align': 'center'}),


        html.Div([


            # Transforms
            html.Div([
                html.Div(["Transforms"]),
                html.Div([html.Div(['Normalization: ', dcc.RadioItems(options=normalization, value=normalization[0],
                                                                      id='normalization-radio-items', inline=True)]),
                          html.Div(['Log Transform: ', dcc.RadioItems(options=log_transform, value=log_transform[0],
                                                                      id='log-transformation-radio-items', inline=True)])
                          ])
            ],
                style={
                    'float': 'left',
                    'display': 'inline-block',
                    'width': '25%',
                }),

            # Graph
            html.Div([
                dcc.Graph(figure={}, id='line-graph')],
                style={
                    'float': 'right',
                    'display': 'inline-block',
                    'width': '75%',
                }
            )],
            style={
                'border': '2px black solid',
                'height': '500px'

            }
        ),



        # Data table
        html.Button('Inspector', id='toggle-inspector', n_clicks=0),
        html.Div([dash_table.DataTable(data=df.to_dict('records'), page_size=10)],
                 id='inspector',
                 style={
                     'display': 'none'
                 })
    ])

    app.run(debug=True)


@callback(
    Output(component_id='line-graph', component_property='figure'),
    Input(component_id='displayed-feature-radio-item', component_property='value')
)
def display_feature(feature):
    fig = px.line(df, x='Date', y=feature)
    return fig


@callback(
   Output(component_id='inspector', component_property='style'),
   Input(component_id='toggle-inspector', component_property='n_clicks'))
def toggle_inspector(n_clicks):
    if n_clicks % 2:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


def normalize_feature(feature):
    pass