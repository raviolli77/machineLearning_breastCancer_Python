import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from urllib.request import urlopen

app = dash.Dash()

# Loading data and cleaning dataset
UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases\
/breast-cancer-wisconsin/wdbc.data'

names = ['id_number', 'diagnosis', 'radius_mean',
         'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean',
         'concavity_mean','concave_points_mean',
         'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se',
         'area_se', 'smoothness_se', 'compactness_se',
         'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se',
         'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst',
         'smoothness_worst', 'compactness_worst',
         'concavity_worst', 'concave_points_worst',
         'symmetry_worst', 'fractal_dimension_worst']

breast_cancer = pd.read_csv(urlopen(UCI_data_URL), names=names)

# Setting 'id_number' as our index
breast_cancer.set_index(['id_number'], inplace = True)

# Converted to binary to help later on with models and plots
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})

#for col in breast_cancer:
	#pd.to_numeric(col, errors='coerce')

app.layout = html.Div(children=[
	html.H1(children='Breast Cancer Dashboard'),

	html.Div(children='''
		An interactive dashboard created by Raul Eulogio
		'''),
	html.Div([
			html.Label('Choose the different parameters'), 
			dcc.Dropdown(
				id='first_input', 
				options=[
				{'label': i, 'value': i} for i in names[2:]
				], 
				value = 'area_worst'
				), 
			dcc.Dropdown(
				id='second_input', 
				options=[
				{'label': i, 'value': i} for i in names[2:]
				], 
				value = 'perimeter_worst'
				),
			dcc.Dropdown(
				id='third_input', 
				options=[
				{'label': i, 'value': i} for i in names[2:]
				], 
				value = 'concave_points_worst'
				)],
		style={'width': '25%', 'display': 'inline-block'}),
	dcc.Graph(
			id='scatter_plot_3d' 
		)
	])

@app.callback(
	dash.dependencies.Output('scatter_plot_3d', 'figure'),
    [dash.dependencies.Input('first_input', 'value'),
    dash.dependencies.Input('second_input', 'value'),
    dash.dependencies.Input('third_input', 'value'),]
    )

def update_figure(first_input_name, second_input_name, third_input_name):
	traces = []
	for i in breast_cancer.diagnosis.unique():
		breast_cancer_dx = breast_cancer[breast_cancer['diagnosis'] == i]
		if (i == 0):
			traces.append(go.Scatter3d(
				x=breast_cancer_dx[first_input_name],
				y=breast_cancer_dx[second_input_name],
				z=breast_cancer_dx[third_input_name],
				text=breast_cancer_dx['diagnosis'],
				mode='markers',
				opacity=0.7,
				marker={
				'size': 15,
				'line': {'width': 0.5, 'color': 'white'},
				'color': 'red'
				},
				name='Malignant'
		))
		else: 
			traces.append(go.Scatter3d(
				x=breast_cancer_dx[first_input_name],
				y=breast_cancer_dx[second_input_name],
				z=breast_cancer_dx[third_input_name],
				text=breast_cancer_dx['diagnosis'],
				mode='markers',
				opacity=0.7,
				marker={
				'size': 15,
				'line': {'width': 0.5, 'color': 'white'},
				'color': '#875FDB'
				},
				name='Benign'
				))
	return {
	'data': traces, 
	'layout': go.Layout(
		xaxis={'title': first_input_name},
		yaxis={'title': second_input_name},
		margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
		legend={'x': 0, 'y': 1},
		hovermode='closest'
		)
	}


if __name__ == '__main__':
    app.run_server(debug=True)