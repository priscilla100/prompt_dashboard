import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import numpy as np
from pathlib import Path

class ExperimentComparator:
    def __init__(self):
        self.experiments = {
            'nl2pl': None,
            'nl2futureltl': None,
            'nl2pastltl': None,
            'textbook_nl2futureltl': None,
            'trace_characterization': None,
            'trace_generation': None,
            'wff': None
        }
        
    def load_data(self, experiment_name, detailed_csv_path, undefined_csv_path):
        """Load both detailed and undefined prompt versions for an experiment"""
        try:
            detailed_df = pd.read_csv(detailed_csv_path)
            undefined_df = pd.read_csv(undefined_csv_path)
            
            # Add version labels
            detailed_df['Prompt_Version'] = 'Well-Defined'
            undefined_df['Prompt_Version'] = 'Undefined'
            
            # Combine datasets
            combined_df = pd.concat([detailed_df, undefined_df], ignore_index=True)
            
            # Clean up the data - handle empty approaches
            combined_df['Approach'] = combined_df['Approach'].fillna('default')
            
            # Create clean labels for x-axis
            combined_df['Model_Clean'] = combined_df['Model'].str.replace('-', ' ').str.title()
            combined_df['Approach_Clean'] = combined_df['Approach'].str.replace('_', ' ').str.title()
            
            self.experiments[experiment_name] = combined_df
            print(f"Loaded {experiment_name}: {len(detailed_df)} well-defined + {len(undefined_df)} undefined samples")
            
        except Exception as e:
            print(f"Error loading {experiment_name}: {e}")
    
    def create_clean_comparison_plot(self, df, metrics, title, height=500):
        """Create a clean comparison plot with overlapping bars"""
        if df is None or df.empty:
            return None
            
        # Create subplot for selected metric
        fig = go.Figure()
        
        # Get unique model-approach combinations
        df['Model_Approach'] = df['Model_Clean'] + '<br>' + df['Approach_Clean']
        unique_combinations = df['Model_Approach'].unique()
        
        # Colors for different versions
        colors = {
            'Well-Defined': 'rgba(31, 119, 180, 0.8)',  # Blue
            'Undefined': 'rgba(255, 127, 14, 0.8)'      # Orange
        }
        
        # Get the first metric as default
        metric = metrics[0] if metrics else list(df.select_dtypes(include=[np.number]).columns)[0]
        
        for version in ['Well-Defined', 'Undefined']:
            version_data = df[df['Prompt_Version'] == version]
            
            fig.add_trace(go.Bar(
                name=version,
                x=version_data['Model_Approach'],
                y=version_data[metric],
                marker_color=colors[version],
                opacity=0.8,
                text=version_data[metric].round(2),
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate=
                '<b>%{x}</b><br>' +
                f'{metric}: %{{y:.2f}}<br>' +
                'Version: ' + version +
                '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f"{title} - {metric}",
                x=0.5,
                font=dict(size=16, family="Arial Black")
            ),
            xaxis_title="Model & Approach",
            yaxis_title=metric,
            barmode='group',
            height=height,
            margin=dict(l=50, r=50, t=80, b=120),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Clean up axes
        fig.update_xaxes(
            tickangle=0,
            tickfont=dict(size=10),
            gridcolor='lightgray',
            gridwidth=0.5
        )
        fig.update_yaxes(
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        )
        
        return fig

# Initialize the comparator
comparator = ExperimentComparator()

# Initialize Dash app with better styling
app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Experiment Results Comparison", 
                style={
                    'textAlign': 'center', 
                    'marginBottom': '10px',
                    'color': '#2c3e50',
                    'fontFamily': 'Arial, sans-serif'
                }),
        html.P("Compare Well-Defined vs Undefined Prompt Performance",
               style={
                   'textAlign': 'center',
                   'color': '#7f8c8d',
                   'fontSize': '16px',
                   'marginBottom': '30px'
               })
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Select Experiment:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='experiment-dropdown',
                options=[
                    {'label': 'NL2PL', 'value': 'nl2pl'},
                    {'label': 'NL2 Future LTL', 'value': 'nl2futureltl'},
                    {'label': 'NL2 Past LTL', 'value': 'nl2pastltl'},
                    {'label': 'Textbook NL2 Future LTL', 'value': 'textbook_nl2futureltl'},
                    {'label': 'WFF Classification', 'value': 'wff'},
                    {'label': 'Trace Generation', 'value': 'trace_generation'},
                    {'label': 'Trace Characterization', 'value': 'trace_characterization'}
                ],
                value='nl2pl',
                style={'marginBottom': '15px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Select Metric:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='metric-dropdown',
                style={'marginBottom': '15px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Chart Height:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Slider(
                id='height-slider',
                min=400,
                max=800,
                step=50,
                value=500,
                marks={i: str(i) for i in range(400, 801, 100)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%', 'verticalAlign': 'top'})
        
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
    
    # Main plot
    html.Div([
        dcc.Graph(id='main-comparison-plot')
    ], style={'padding': '20px'}),
    
    # Summary statistics
    html.Div([
        html.H3("Summary Statistics", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div(id='summary-stats')
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginTop': '20px'}),
    
    # Instructions
    html.Div([
        html.H3("Loading Instructions", style={'color': '#2c3e50'}),
        html.P("Load your data using:", style={'marginBottom': '10px'}),
        html.Pre("""
# Load individual experiments
comparator.load_data('nl2pl', 'detailed_nl2pl.csv', 'undefined_nl2pl.csv')
comparator.load_data('nl2futureltl', 'detailed_nl2futureltl.csv', 'undefined_nl2futureltl.csv')

# Then run: app.run_server(debug=True)
        """, style={
            'backgroundColor': '#2c3e50', 
            'color': 'white',
            'padding': '15px', 
            'borderRadius': '5px',
            'fontSize': '12px'
        })
    ], style={'padding': '20px', 'marginTop': '20px'})
])

# Callback to update metric dropdown based on selected experiment
@app.callback(
    Output('metric-dropdown', 'options'),
    Output('metric-dropdown', 'value'),
    Input('experiment-dropdown', 'value')
)
def update_metric_dropdown(selected_experiment):
    if selected_experiment not in comparator.experiments or comparator.experiments[selected_experiment] is None:
        return [], None
    
    df = comparator.experiments[selected_experiment]
    
    # Define metric mappings for each experiment type
    metric_mappings = {
        'nl2pl': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision', 
            'Recall': 'Recall',
            'F1 Score': 'F1',
            'Jaccard Index': 'Jaccard',
            'Levenshtein Distance': 'Levenshtein'
        },
        'nl2futureltl': {
            'GT→Pred Accuracy': 'Accuracy_GT_to_Pred (%)',
            'Pred→GT Accuracy': 'Accuracy_Pred_to_GT (%)',
            'Equivalence Accuracy': 'Equivalence_Accuracy (%)',
            'Syntactic Correctness': 'Syntactic_Correctness_Rate (%)',
            'Syntactic Match': 'Syntactic_Match_Rate (%)',
            'Precision': 'Precision (%)',
            'Recall': 'Recall (%)',
            'F1 Score': 'F1 (%)'
        },
        'wff': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision',
            'Recall': 'Recall',
            'F1 Score': 'F1_Score'
        },
        'trace_generation': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision',
            'Recall': 'Recall',
            'F1 Score': 'F1_Score',
            'Positive Satisfaction Rate': 'Positive_Satisfaction_Rate',
            'Negative Falsification Rate': 'Negative_Falsification_Rate'
        },
        'trace_characterization': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision',
            'F1 Score': 'F1'
        }
    }
    
    # Use the same mappings for similar experiments
    if selected_experiment in ['nl2pastltl', 'textbook_nl2futureltl']:
        mapping = metric_mappings['nl2futureltl']
    else:
        mapping = metric_mappings.get(selected_experiment, {})
    
    # Filter available metrics
    available_metrics = []
    for display_name, column_name in mapping.items():
        if column_name in df.columns:
            available_metrics.append({'label': display_name, 'value': column_name})
    
    # If no predefined mapping, use numeric columns
    if not available_metrics:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_metrics = [{'label': col, 'value': col} for col in numeric_cols]
    
    default_value = available_metrics[0]['value'] if available_metrics else None
    
    return available_metrics, default_value

# Main callback for updating the plot
@app.callback(
    Output('main-comparison-plot', 'figure'),
    Output('summary-stats', 'children'),
    Input('experiment-dropdown', 'value'),
    Input('metric-dropdown', 'value'),
    Input('height-slider', 'value')
)
def update_main_plot(selected_experiment, selected_metric, chart_height):
    if (selected_experiment not in comparator.experiments or 
        comparator.experiments[selected_experiment] is None or 
        selected_metric is None):
        
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data loaded for this experiment.<br>Please load your CSV files first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        empty_fig.update_layout(height=400, plot_bgcolor='white')
        
        return empty_fig, html.P("No data available", style={'color': 'gray'})
    
    df = comparator.experiments[selected_experiment]
    
    # Create the comparison plot
    fig = comparator.create_clean_comparison_plot(
        df, [selected_metric], 
        selected_experiment.upper().replace('_', ' '),
        height=chart_height
    )
    
    # Update the figure to show the selected metric
    if fig and selected_metric in df.columns:
        # Clear existing traces
        fig.data = []
        
        # Add new traces for the selected metric
        colors = {
            'Well-Defined': 'rgba(52, 152, 219, 0.8)',  # Blue
            'Undefined': 'rgba(231, 76, 60, 0.8)'       # Red
        }
        
        df['Model_Approach'] = df['Model_Clean'] + '<br>' + df['Approach_Clean']
        
        for version in ['Well-Defined', 'Undefined']:
            version_data = df[df['Prompt_Version'] == version]
            
            fig.add_trace(go.Bar(
                name=version,
                x=version_data['Model_Approach'],
                y=version_data[selected_metric],
                marker_color=colors[version],
                opacity=0.8,
                text=version_data[selected_metric].round(2),
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate=
                '<b>%{x}</b><br>' +
                f'{selected_metric}: %{{y:.2f}}<br>' +
                'Version: ' + version +
                '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{selected_experiment.upper().replace('_', ' ')} - {selected_metric}",
            yaxis_title=selected_metric,
            height=chart_height
        )
    
    # Generate summary statistics
    summary_stats = generate_summary_stats(df, selected_metric)
    
    return fig, summary_stats

def generate_summary_stats(df, metric):
    """Generate summary statistics for the selected metric"""
    if df is None or metric not in df.columns:
        return html.P("No statistics available", style={'color': 'gray'})
    
    stats = []
    
    for version in ['Well-Defined', 'Undefined']:
        version_data = df[df['Prompt_Version'] == version]
        if not version_data.empty:
            mean_val = version_data[metric].mean()
            std_val = version_data[metric].std()
            max_val = version_data[metric].max()
            min_val = version_data[metric].min()
            
            stats.append(
                html.Div([
                    html.H4(f"{version} Prompts", style={'color': '#3498db' if version == 'Well-Defined' else '#e74c3c'}),
                    html.P(f"Mean: {mean_val:.3f} ± {std_val:.3f}"),
                    html.P(f"Range: {min_val:.3f} - {max_val:.3f}"),
                    html.P(f"Samples: {len(version_data)}")
                ], style={'display': 'inline-block', 'width': '45%', 'marginRight': '5%', 'verticalAlign': 'top'})
            )
    
    # Calculate improvement
    if len(stats) == 2:
        well_defined_mean = df[df['Prompt_Version'] == 'Well-Defined'][metric].mean()
        undefined_mean = df[df['Prompt_Version'] == 'Undefined'][metric].mean()
        
        if metric == 'Levenshtein':  # Lower is better for Levenshtein
            improvement = ((undefined_mean - well_defined_mean) / undefined_mean) * 100
            direction = "reduction" if improvement > 0 else "increase"
        else:  # Higher is better for most metrics
            improvement = ((well_defined_mean - undefined_mean) / undefined_mean) * 100
            direction = "improvement" if improvement > 0 else "decline"
        
        stats.append(
            html.Div([
                html.H4("Comparison", style={'color': '#2c3e50'}),
                html.P(f"{improvement:.1f}% {direction} with well-defined prompts",
                       style={'fontWeight': 'bold', 
                              'color': 'green' if improvement > 0 else 'red'})
            ], style={'display': 'inline-block', 'width': '45%', 'verticalAlign': 'top'})
        )
    
    return html.Div(stats)

# Utility functions (same as before)
def load_all_experiments(base_path="./"):
    """Utility function to load all experiments at once"""
    experiments = ['nl2pl', 'nl2futureltl', 'nl2pastltl', 'textbook_nl2futureltl', 
                  'trace_characterization', 'trace_generation', 'wff']
    
    for exp in experiments:
        detailed_path = f"{base_path}/detailed_{exp}.csv"
        undefined_path = f"{base_path}/undefined_{exp}.csv"
        
        if Path(detailed_path).exists() and Path(undefined_path).exists():
            comparator.load_data(exp, detailed_path, undefined_path)
        else:
            print(f"Files not found for {exp}: {detailed_path}, {undefined_path}")

if __name__ == '__main__':
    # Example data loading - uncomment and modify paths as needed
    # comparator.load_data('nl2pl', 'detailed_nl2pl.csv', 'undefined_nl2pl.csv')
    # load_all_experiments("path/to/your/csv/files/")
    
    app.run_server(debug=True)
