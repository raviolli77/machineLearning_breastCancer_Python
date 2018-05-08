#!/usr/bin/env python3

import sys
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import global_vars as gv
import pandas as pd

sys.path.insert(0, '../src/python/')
from data_extraction import breast_cancer, names
sys.path.pop(0)

# Test set metrics
cross_tab_knn = gv.cross_tab_knn
cross_tab_rf = gv.cross_tab_rf
cross_tab_nn = gv.cross_tab_nn

# Classification Reports
class_rep_knn = gv.class_rep_knn
class_rep_rf = gv.class_rep_rf
class_rep_nn = gv.class_rep_nn

def generate_table(dataframe, max_rows=10):
	return html.Table(
		# Header
		[html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
		[html.Tr([
			html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
		]) for i in range(min(len(dataframe), max_rows))]
	)

app = dash.Dash()

app.layout = html.Div([
	html.Div([
        html.H2("Breast Cancer Dashboard"),
        ], className='banner'),
	html.H2(children = '''
		An interactive dashboard created by Raul Eulogio
		''',
        style={
        'padding': '0px 30px 15px 30px'}),
    html.Div([
        html.H3(children = '''
            Exploratory Analysis
            ''',
            style={
            'padding': '0px 30px 15px 30px'})
        ]),
	html.Div([
			html.Div([
					html.P("""
						Move the multi-select options to see the 3d scatter plot and histograms change respectively.
						And play with the interactive 3d scatter plot to see how variables interact!

						"""),
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
						),
					dcc.Graph(
						id='scatter_plot_3d'),
					html.Div(html.P(' .')),
				html.Div([
					html.H3("""
					Machine Learning
					"""),
					dcc.Markdown('Here are some metrics relating to how well each model did.'),
					dcc.Markdown('+ See [this article](https://lukeoakdenrayner.wordpress.com/2017/12/06/do-machines-actually-beat-doctors-roc-curves-and-performance-metrics/) for more information about *ROC Curves* '),
					html.Label('Choose a Machine Learning Model'),
						dcc.Dropdown(
    						id='machine_learning',
    						options=[
    						{'label': 'Kth Nearest Neighor', 'value': 'knn'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'Neural Network', 'value': 'nn'}
    						],
    						value = 'knn'
    						),
						dcc.Graph(
							id='roc_curve')
						])
					],
					style={'width': '40%',
                    'height': '50%',
					'float': 'left',
					'padding': '0px 40px 40px 40px'}),
			# End Left Side Div
			# Right Side Div
			html.Div([
				dcc.Graph(
					id='hist_first_var',
					style={'height': '12%'}
					),
				dcc.Graph(
					id='hist_sec_var',
					style={'height': '12%'}
					),
				dcc.Graph(
					id='hist_third_var',
					style={'height': '12%'}
					),
				html.Div(html.P(' .')),
				html.Div(html.P(' .')),
				html.Div(html.P(' .')),
				html.Div(html.P(' .')),
				html.Div(
					html.H4("""
					Test Set Metrics
					"""
					)
					),
				dcc.Markdown("+ See [Test Set Metrics Section of inertia7 project](https://www.inertia7.com/projects/95#test_set_met) for more information."),
				html.Div(
					dcc.Graph(
						id="conf_mat",
						style={'height': '10%'}
						)
					),
				html.Div(
					html.H4("""
					Classification Report
					"""
					)),
				dcc.Markdown("+ See [Classification Report Section of inertia7 project](https://www.inertia7.com/projects/95) for more information. "),
				html.Div([html.Div(id='table_class_rep')
					],
					style={'width': '100%'})
				],
				style={'width': '40%',
				'float': 'right',
				'padding': '0px 40px 40px 40px'},
				)
			# End Right Side Div
			],
			style={'width': '100%',
			'height': '100%',
			'display': 'flex'}),
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


@app.callback(
	dash.dependencies.Output('hist_first_var', 'figure'),
	[dash.dependencies.Input('first_input', 'value')]
	)
def update_hist_1(first_input_name):
	traces_hist = []
	for i in breast_cancer.diagnosis.unique():
		breast_cancer_dx = breast_cancer[breast_cancer['diagnosis'] == i]
		if (i == 0):
			traces_hist.append(go.Histogram(
				x = breast_cancer_dx[first_input_name],
				opacity=0.60,
				marker={
				'color': 'red'
				},
				name='Malignant'
				))
		else:
			traces_hist.append(go.Histogram(
				x = breast_cancer_dx[first_input_name],
				opacity=0.60,
				marker={
				'color': '#875FDB'
				},
				name='Benign',
				))
	return {
	'data': traces_hist,
	'layout': go.Layout(
		xaxis={'title': first_input_name},
		margin={'l': 50, 'b': 40, 't': 10, 'r': 10},
		legend={'x': 0, 'y': 1},
		hovermode='closest',
		barmode='overlay'
		)
	}

@app.callback(
	dash.dependencies.Output('hist_sec_var', 'figure'),
   	[dash.dependencies.Input('second_input', 'value')]
   	)
def update_hist_2(second_input):
	traces_hist = []
	for i in breast_cancer.diagnosis.unique():
		breast_cancer_dx = breast_cancer[breast_cancer['diagnosis'] == i]
		if (i == 0):
			traces_hist.append(go.Histogram(
				x = breast_cancer_dx[second_input],
				opacity=0.60,
				marker={
				'color': 'red'
				},
				name='Malignant'
				))
		else:
			traces_hist.append(go.Histogram(
				x = breast_cancer_dx[second_input],
				opacity=0.60,
				marker={
				'color': '#875FDB'
				},
				name='Benign',
				))
	return {
	'data': traces_hist,
	'layout': go.Layout(
		xaxis={'title': second_input},
		margin={'l': 50, 'b': 40, 't': 10, 'r': 10},
		legend={'x': 0, 'y': 1},
		hovermode='closest',
		barmode='overlay'
		)
	}

@app.callback(
	dash.dependencies.Output('hist_third_var', 'figure'),
	[dash.dependencies.Input('third_input', 'value')]
	)
def update_hist_3(third_input):
	traces_hist = []
	for i in breast_cancer.diagnosis.unique():
		breast_cancer_dx = breast_cancer[breast_cancer['diagnosis'] == i]
		if (i == 0):
			traces_hist.append(go.Histogram(
				x = breast_cancer_dx[third_input],
				opacity=0.60,
				marker={
				'color': 'red'
				},
				name='Malignant'
				))
		else:
			traces_hist.append(go.Histogram(
				x = breast_cancer_dx[third_input],
				opacity=0.60,
				marker={
				'color': '#875FDB'
				},
				name='Benign',
				))
	return {
	'data': traces_hist,
	'layout': go.Layout(
		xaxis={'title': third_input},
		margin={'l': 50, 'b': 40, 't': 10, 'r': 10},
		legend={'x': 0, 'y': 1},
		hovermode='closest',
		barmode='overlay'
		)
	}


@app.callback(
	dash.dependencies.Output('roc_curve', 'figure'),
    [dash.dependencies.Input('machine_learning', 'value')
    ])

def update_roc(machine_learning):
	lw = 2
	if (machine_learning == 'knn'):
		trace1 = go.Scatter(
			x = gv.fpr, y = gv.tpr,
			mode='lines',
			line=dict(color='deeppink', width=lw),
			name='ROC curve (AUC = {0: 0.3f})'.format(gv.auc_knn))
	if (machine_learning == 'rf'):
		trace1 = go.Scatter(
			x = gv.fpr2, y = gv.tpr2,
			mode='lines',
			line=dict(color='red', width=lw),
			name='ROC curve (AUC = {0: 0.3f})'.format(gv.auc_rf))
	if (machine_learning == 'nn'):
		trace1 = go.Scatter(
			x = gv.fpr3, y = gv.tpr3,
			mode='lines',
			line=dict(color='purple', width=lw),
			name='ROC curve (AUC = {0: 0.3f})'.format(gv.auc_nn))
	trace2 = go.Scatter(x=[0, 1], y=[0, 1],
			mode='lines',
			line=dict(color='black', width=lw, dash='dash'),
			showlegend=False)
	trace3 = go.Scatter(x=[0, 0], y=[1, 0],
			mode='lines',
			line=dict(color='black', width=lw, dash='dash'),
			showlegend=False)
	trace4 = go.Scatter(x=[1, 0], y=[1, 1],
			mode='lines',
			line=dict(color='black', width=lw, dash='dash'),
			showlegend=False)
	return {
	'data': [trace1, trace2, trace3, trace4],
	'layout': go.Layout(
		title='Receiver Operating Characteristic Plot',
        xaxis={'title': 'False Positive Rate'},
        yaxis={'title': 'True Positive Rate'},
        legend={'x': 0.7, 'y': 0.15},
        #height=400
        )
	}

@app.callback(
	dash.dependencies.Output('conf_mat', 'figure'),
    [dash.dependencies.Input('machine_learning', 'value')
    ])

def update_conf_mat(machine_learning):
	lw = 2
	if (machine_learning == 'knn'):
		trace1 = go.Heatmap(
			z = np.roll(cross_tab_knn,
				1, axis=0))
	if (machine_learning == 'rf'):
		trace1 = go.Heatmap(
			z = np.roll(cross_tab_rf,
				1, axis=0))
	if (machine_learning == 'nn'):
		trace1 = go.Heatmap(
			z = np.roll(cross_tab_nn,
				1, axis=0))
	return {
	'data': [trace1],
	'layout': go.Layout(
		title='Confusion Matrix',
        xaxis={'title': 'Predicted Values'},
        yaxis={'title': 'Actual Values'}
        )
	}

####################################
#
#
#
#def update_table(machine_learning):
	#final_cross_tab = pd.DataFrame()
	#if (machine_learning == 'knn'):
		#final_cross_tab = cross_tab_knn
	#if (machine_learning == 'rf'):
		#final_cross_tab = cross_tab_rf
	#if (machine_learning == 'nn'):
		#final_cross_tab = cross_tab_nn
	#return generate_table(dataframe = final_cross_tab)


@app.callback(
	dash.dependencies.Output('table_class_rep', 'children'),
    [dash.dependencies.Input('machine_learning', 'value')
    ])
def update_table(machine_learning):
	final_cross_tab = pd.DataFrame()
	if (machine_learning == 'knn'):
		final_cross_tab = class_rep_knn
	if (machine_learning == 'rf'):
		final_cross_tab = class_rep_rf
	if (machine_learning == 'nn'):
		final_cross_tab = class_rep_nn
	return generate_table(dataframe = final_cross_tab)


# Append externally hosted CSS Stylesheet
my_css_urls = [
# For dev:
'https://rawgit.com/raviolli77/machineLearning_breastCancer_Python/master/dash_dashboard/dash_breast_cancer.css',
# For prod
#'https://cdn.rawgit.com/raviolli77/machineLearning_breastCancer_Python/master/dash_dashboard/dash_breast_cancer.css'
]

app.css.append_css({
	'external_url': my_css_urls
	})

if __name__ == '__main__':
    app.run_server(debug=True)
