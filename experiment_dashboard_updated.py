import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import numpy as np
from pathlib import Path
import os
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
            undefined_df['Prompt_Version'] = 'Not Fully-Defined'
            
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
        """Create a clean comparison plot with better x-axis organization"""
        if df is None or df.empty:
            return None
        
        # Get the first metric as default
        metric = metrics[0] if metrics else list(df.select_dtypes(include=[np.number]).columns)[0]
        
        # Create a more organized structure for x-axis
        df_plot = df.copy()
        
        # Create cleaner labels
        df_plot['Model_Short'] = df_plot['Model_Clean'].str.replace('GPT-', 'GPT').str.replace('Claude-', 'Claude')
        df_plot['Approach_Short'] = df_plot['Approach_Clean'].str.replace('Approach', '').str.strip()
        
        # Create a hierarchical x-axis structure
        df_plot['Primary_Label'] = df_plot['Model_Short']
        df_plot['Secondary_Label'] = df_plot['Approach_Short']
        
        # Sort by model and approach for better organization
        df_plot = df_plot.sort_values(['Model_Short', 'Approach_Short'])
        
        # Create subplot with secondary x-axis for better organization
        fig = make_subplots(
            rows=1, cols=1,
            # subplot_titles=[f"{title} - {metric}"],
            specs=[[{"secondary_y": False}]]
        )
        
        # Enhanced color scheme
        colors = {
            'Well-Defined': '#3498db',      # Professional blue
            'Not Fully-Defined': '#e74c3c'  # Professional red
        }
        
        # Get unique combinations for proper spacing
        unique_combinations = df_plot.groupby(['Model_Short', 'Approach_Short']).size().reset_index()
        x_positions = range(len(unique_combinations))
        
        # Create labels for x-axis
        x_labels = []
        x_tick_positions = []
        
        for i, (_, row) in enumerate(unique_combinations.iterrows()):
            model = row['Model_Short']
            approach = row['Approach_Short']
            x_labels.append(f"{model}<br><sub>{approach}</sub>")
            x_tick_positions.append(i)
        
        # Add bars for each version
        bar_width = 0.35
        for i, version in enumerate(['Well-Defined', 'Not Fully-Defined']):
            version_data = df_plot[df_plot['Prompt_Version'] == version]
            
            # Match data to x-positions
            y_values = []
            text_values = []
            
            for _, combo_row in unique_combinations.iterrows():
                model = combo_row['Model_Short']
                approach = combo_row['Approach_Short']
                
                matching_data = version_data[
                    (version_data['Model_Short'] == model) & 
                    (version_data['Approach_Short'] == approach)
                ]
                
                if not matching_data.empty:
                    value = matching_data[metric].iloc[0]
                    y_values.append(value)
                    text_values.append(f"{value:.2f}")
                else:
                    y_values.append(0)
                    text_values.append("N/A")
            
            # Calculate x positions for grouped bars
            x_pos = [pos + (i - 0.5) * bar_width for pos in x_tick_positions]
            
            fig.add_trace(go.Bar(
                name=version,
                x=x_pos,
                y=y_values,
                marker_color=colors[version],
                marker_line=dict(width=1, color='white'),
                opacity=0.8,
                text=text_values,
                textposition='outside',
                textfont=dict(size=10, color='black'),
                width=bar_width,
                hovertemplate=
                    '<b>%{customdata[0]} - %{customdata[1]}</b><br>' +
                    f'{metric}: %{{y:.2f}}<br>' +
                    'Version: ' + version +
                    '<extra></extra>',
                customdata=[[combo_row['Model_Short'], combo_row['Approach_Short']] 
                        for _, combo_row in unique_combinations.iterrows()]
            ))
        
        # Update layout with improved styling
        fig.update_layout(
    # title=dict(
    #     text=f"{title} - {metric}",
    #     x=0.5,
    #     font=dict(size=18, family="Arial", color='#2c3e50')
    # ),
    xaxis=dict(
        title=dict( # Corrected: Title object for x-axis
            text="Model & Approach",
            font=dict(size=14, color='#34495e')
        ),
        tickmode='array',
        tickvals=x_tick_positions,
        ticktext=x_labels,
        tickfont=dict(size=11, color='#34495e'),
        gridcolor='rgba(128, 128, 128, 0.2)',
        gridwidth=1,
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        title=dict( # Corrected: Title object for y-axis
            text=metric,
            font=dict(size=14, color='#34495e')
        ),
        tickfont=dict(size=11, color='#34495e'),
        gridcolor='rgba(128, 128, 128, 0.2)',
        gridwidth=1,
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(128, 128, 128, 0.3)',
        zerolinewidth=1
    ),
            barmode='group',
            height=height,
            margin=dict(l=60, r=60, t=100, b=80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color='#34495e'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(128, 128, 128, 0.3)',
                borderwidth=1
            ),
            font=dict(size=12, family="Arial", color='#2c3e50'),
            plot_bgcolor='rgba(248, 249, 250, 1)',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig

    def create_faceted_comparison_plot(self, df, metrics, title, height=600):
        """Alternative: Create a faceted plot for even cleaner separation"""
        if df is None or df.empty:
            return None
        
        metric = metrics[0] if metrics else list(df.select_dtypes(include=[np.number]).columns)[0]
        
        # Prepare data
        df_plot = df.copy()
        df_plot['Model_Short'] = df_plot['Model_Clean'].str.replace('GPT-', 'GPT').str.replace('Claude-', 'Claude')
        
        # Get unique models for subplots
        unique_models = sorted(df_plot['Model_Short'].unique())
        
        # Create subplots - one for each model
        fig = make_subplots(
            rows=1, 
            cols=len(unique_models),
            subplot_titles=unique_models,
            shared_yaxes=True,
            horizontal_spacing=0.05
        )
        
        colors = {
            'Well-Defined': '#3498db',
            'Not Fully-Defined': '#e74c3c'
        }
        
        for col, model in enumerate(unique_models, 1):
            model_data = df_plot[df_plot['Model_Short'] == model]
            
            for version in ['Well-Defined', 'Not Fully-Defined']:
                version_data = model_data[model_data['Prompt_Version'] == version]
                
                fig.add_trace(go.Bar(
                    name=version,
                    x=version_data['Approach_Clean'],
                    y=version_data[metric],
                    marker_color=colors[version],
                    opacity=0.8,
                    text=version_data[metric].round(2),
                    textposition='outside',
                    textfont=dict(size=10),
                    showlegend=(col == 1),  # Show legend only for first subplot
                    hovertemplate=
                        f'<b>{model} - %{{x}}</b><br>' +
                        f'{metric}: %{{y:.2f}}<br>' +
                        'Version: ' + version +
                        '<extra></extra>'
                ), row=1, col=col)
        
        fig.update_layout(
            title=dict(
                text=f"{title} - {metric}",
                x=0.5,
                font=dict(size=18, family="Arial", color='#2c3e50')
            ),
            height=height,
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor='rgba(248, 249, 250, 1)',
            paper_bgcolor='white'
        )
        
        # Update all x-axes
        for i in range(1, len(unique_models) + 1):
            fig.update_xaxes(
                title_text="Approach" if i == len(unique_models)//2 + 1 else "",
                tickangle=45,
                row=1, col=i
            )
        
        # Update y-axis
        fig.update_yaxes(title_text=metric, row=1, col=1)
        
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
        html.P("Compare Well-Defined vs Not Fully-Defined Prompt Performance",
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
            'Not Fully-Defined': 'rgba(231, 76, 60, 0.8)'       # Red
        }
        
        df['Model_Approach'] = df['Model_Clean'] + '<br>' + df['Approach_Clean']
        
        for version in ['Well-Defined', 'Not Fully-Defined']:
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
    
    for version in ['Well-Defined', 'Not Fully-Defined']:
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
        undefined_mean = df[df['Prompt_Version'] == 'Not Fully-Defined'][metric].mean()
        
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
def export_comparison_images(output_dir="comparison_images"):
        os.makedirs(output_dir, exist_ok=True)
        for exp_name, df in comparator.experiments.items():
            if df is not None and not df.empty:
                # Use the first available metric for each experiment
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    continue
                metric = numeric_cols[0]
                fig = comparator.create_clean_comparison_plot(df, [metric], exp_name.upper())
                if fig:
                    out_path = os.path.join(output_dir, f"{exp_name}_{metric}.png")
                    fig.write_image(out_path)
                    print(f"Exported {out_path}")
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

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    defined_nl2futureltl_little_tricky_path = os.path.join(script_dir, 'data', 'defined', 'comprehensive_table_future_little_tricky.csv')
    defined_nl2pl_path = os.path.join(script_dir, 'data', 'defined', 'nl2pl_aggregated_results.csv')
    defined_wff_path = os.path.join(script_dir, 'data', 'defined', 'wff_aggregate_metrics.csv')
    defined_nl2future_textbook_path = os.path.join(script_dir, 'data', 'defined', 'comprehensive_table_future_textbook.csv')
    defined_nl2pastltl_little_tricky_path = os.path.join(script_dir, 'data', 'defined', 'comprehensive_table_future_textbook.csv')
    defined_trace_characterization_path = os.path.join(script_dir, 'data', 'defined', 'trace_characterization.csv')
    defined_trace_generation_path = os.path.join(script_dir, 'data', 'defined', 'trace_generation.csv')

    # For undefined data:
    undefined_nl2futureltl_little_tricky_path = os.path.join(script_dir, 'data', 'undefined', 'comprehensive_table_future_little_tricky.csv')
    undefined_nl2pl_path = os.path.join(script_dir, 'data', 'undefined', 'nl2pl_aggregated_results.csv')
    undefined_wff_path = os.path.join(script_dir, 'data', 'undefined', 'wff_aggregate_metrics.csv')
    undefined_nl2future_textbook_path = os.path.join(script_dir, 'data', 'undefined', 'comprehensive_table_future_textbook.csv')
    undefined_nl2pastltl_little_tricky_path = os.path.join(script_dir, 'data', 'undefined', 'comprehensive_table_future_textbook.csv')
    undefined_trace_characterization_path = os.path.join(script_dir, 'data', 'undefined', 'trace_characterization.csv')
    undefined_trace_generation_path = os.path.join(script_dir, 'data', 'undefined', 'trace_generation.csv')

    # Load defined and undefined data
    try:
        comparator.load_data('nl2futureltl',
                             defined_nl2futureltl_little_tricky_path,
                             undefined_nl2futureltl_little_tricky_path)
        print(f"Loaded nl2futureltl from: {defined_nl2futureltl_little_tricky_path} and {undefined_nl2futureltl_little_tricky_path}")
    except FileNotFoundError as e:
        print(f"Error loading nl2futureltl: {e}")

    try:
        comparator.load_data('nl2pl',
                             defined_nl2pl_path,
                             undefined_nl2pl_path)
        print(f"Loaded nl2pl from: {defined_nl2pl_path} and {undefined_nl2pl_path}")
    except FileNotFoundError as e:
        print(f"Error loading wff from: {e}")

    try:
        comparator.load_data('wff',
                             defined_wff_path,
                             undefined_wff_path)
        print(f"Loaded wff from: {defined_wff_path} and {undefined_wff_path}")
    except FileNotFoundError as e:
        print(f"Error loading wff from: {e}")

    try:
        comparator.load_data('textbook_nl2futureltl',
                             defined_nl2future_textbook_path,
                             undefined_nl2future_textbook_path)
        print(f"Loaded wff from: {defined_nl2future_textbook_path} and {undefined_nl2future_textbook_path}")
    except FileNotFoundError as e:
        print(f"Error loading wff from: {e}")

    # Load other experiments as needed
    # Load your data here
    # comparator.load_data('nl2futureltl', 
    #                      'dashboard/data/defined/comprehensive_table.csv',
    #                      'dashboard/data/undefined/comprehensive_table.csv')
    # comparator.load_data('nl2pl', 
    #                      'dashboard/data/defined/nl2pl_aggregated_results.csv',
    #                      'dashboard/data/undefined/nl2pl_aggregated_results.csv')
    # comparator.load_data('wff', 
    #                      'dashboard/data/defined/wff_aggregate_metrics.csv',
    #                      'dashboard/data/undefined/wff_aggregate_metrics.csv')
    
    # Or use the utility function:
    # load_all_experiments("path/to/your/csv/files/")
    
    # Export static images if needed:
    # export_comparison_images()
    
    
    # Export static images if needed:
    # Define export_comparison_images if you want to export static images of all loaded experiments

    

    # Uncomment to export images:
    export_comparison_images()
    app.run(debug=True, port=8199)

