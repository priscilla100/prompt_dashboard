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
            detailed_df['Prompt_Version'] = 'Detailed'
            undefined_df['Prompt_Version'] = 'Undefined'
            
            # Combine datasets
            combined_df = pd.concat([detailed_df, undefined_df], ignore_index=True)
            
            self.experiments[experiment_name] = combined_df
            print(f"Loaded {experiment_name}: {len(detailed_df)} detailed + {len(undefined_df)} undefined samples")
            
        except Exception as e:
            print(f"Error loading {experiment_name}: {e}")
    
    def create_nl2pl_comparison(self):
        """Create comparison plots for NL2PL experiment"""
        if self.experiments['nl2pl'] is None:
            return None
            
        df = self.experiments['nl2pl']
        
        # Metrics to compare
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Jaccard']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics + ['Levenshtein Distance'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = {'Detailed': '#1f77b4', 'Undefined': '#ff7f0e'}
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            for version in ['Detailed', 'Undefined']:
                version_data = df[df['Prompt_Version'] == version]
                fig.add_trace(
                    go.Bar(
                        x=version_data['Model'] + ' - ' + version_data['Approach'],
                        y=version_data[metric],
                        name=f'{version} - {metric}',
                        marker_color=colors[version],
                        opacity=0.7,
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        # Levenshtein distance (lower is better)
        for version in ['Detailed', 'Undefined']:
            version_data = df[df['Prompt_Version'] == version]
            fig.add_trace(
                go.Bar(
                    x=version_data['Model'] + ' - ' + version_data['Approach'],
                    y=version_data['Levenshtein'],
                    name=f'{version} - Levenshtein',
                    marker_color=colors[version],
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title="NL2PL Experiment: Detailed vs Undefined Prompts",
            height=800,
            showlegend=True
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_ltl_comparison(self, experiment_names):
        """Create comparison for LTL experiments (nl2futureltl, nl2pastltl, textbook_nl2futureltl)"""
        ltl_data = []
        
        for exp_name in experiment_names:
            if self.experiments[exp_name] is not None:
                df = self.experiments[exp_name].copy()
                df['Experiment'] = exp_name
                ltl_data.append(df)
        
        if not ltl_data:
            return None
            
        combined_df = pd.concat(ltl_data, ignore_index=True)
        
        # Key metrics for LTL
        metrics = ['Accuracy_GT_to_Pred (%)', 'Accuracy_Pred_to_GT (%)', 
                  'Equivalence_Accuracy (%)', 'Syntactic_Correctness_Rate (%)',
                  'Precision (%)', 'Recall (%)', 'F1 (%)']
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=metrics + ['Syntactic Match Rate (%)', 'Model Performance Summary'],
            specs=[[{"secondary_y": False}] * 3] * 3
        )
        
        colors = {'Detailed': '#2E8B57', 'Undefined': '#DC143C'}
        
        for i, metric in enumerate(metrics):
            if i >= 8:  # Skip if we have more metrics than subplots
                break
                
            row = i // 3 + 1
            col = i % 3 + 1
            
            metric_data = []
            for _, row_data in combined_df.iterrows():
                metric_data.append({
                    'Experiment': row_data['Experiment'],
                    'Model_Approach': f"{row_data['Model']}-{row_data['Approach']}",
                    'Version': row_data['Prompt_Version'],
                    'Value': row_data[metric] if metric in row_data else 0
                })
            
            metric_df = pd.DataFrame(metric_data)
            
            for version in ['Detailed', 'Undefined']:
                version_data = metric_df[metric_df['Version'] == version]
                fig.add_trace(
                    go.Bar(
                        x=version_data['Experiment'] + ' - ' + version_data['Model_Approach'],
                        y=version_data['Value'],
                        name=f'{version}',
                        marker_color=colors[version],
                        opacity=0.7,
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        # Add Syntactic Match Rate
        if 'Syntactic_Match_Rate (%)' in combined_df.columns:
            for version in ['Detailed', 'Undefined']:
                version_data = combined_df[combined_df['Prompt_Version'] == version]
                fig.add_trace(
                    go.Bar(
                        x=version_data['Experiment'] + ' - ' + version_data['Model'] + '-' + version_data['Approach'],
                        y=version_data['Syntactic_Match_Rate (%)'],
                        name=f'{version}',
                        marker_color=colors[version],
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=3, col=2
                )
        
        fig.update_layout(
            title="LTL Experiments Comparison: Detailed vs Undefined Prompts",
            height=1200,
            showlegend=True
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_wff_comparison(self):
        """Create comparison for WFF experiment"""
        if self.experiments['wff'] is None:
            return None
            
        df = self.experiments['wff']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1 Score']
        )
        
        colors = {'Detailed': '#9370DB', 'Undefined': '#FF6347'}
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            for version in ['Detailed', 'Undefined']:
                version_data = df[df['Prompt_Version'] == version]
                fig.add_trace(
                    go.Bar(
                        x=version_data['Model'] + ' - ' + version_data['Approach'],
                        y=version_data[metric],
                        name=f'{version}',
                        marker_color=colors[version],
                        opacity=0.7,
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="WFF Experiment: Detailed vs Undefined Prompts",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_trace_experiments_comparison(self):
        """Create comparison for trace generation and characterization"""
        trace_gen = self.experiments['trace_generation']
        trace_char = self.experiments['trace_characterization']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Trace Gen: Accuracy', 'Trace Gen: Precision', 'Trace Gen: F1',
                           'Trace Char: Accuracy', 'Trace Char: Precision', 'Trace Char: F1'],
            specs=[[{"secondary_y": False}] * 3] * 2
        )
        
        colors = {'Detailed': '#20B2AA', 'Undefined': '#F4A460'}
        
        # Trace Generation plots
        if trace_gen is not None:
            metrics = ['Accuracy', 'Precision', 'F1_Score']
            for i, metric in enumerate(metrics):
                for version in ['Detailed', 'Undefined']:
                    version_data = trace_gen[trace_gen['Prompt_Version'] == version]
                    fig.add_trace(
                        go.Bar(
                            x=version_data['Model'] + ' - ' + version_data['Approach'],
                            y=version_data[metric],
                            name=f'{version}',
                            marker_color=colors[version],
                            opacity=0.7,
                            showlegend=(i == 0)
                        ),
                        row=1, col=i+1
                    )
        
        # Trace Characterization plots
        if trace_char is not None:
            metrics = ['Accuracy', 'Precision', 'F1']
            for i, metric in enumerate(metrics):
                for version in ['Detailed', 'Undefined']:
                    version_data = trace_char[trace_char['Prompt_Version'] == version]
                    fig.add_trace(
                        go.Bar(
                            x=version_data['Model'] + ' - ' + version_data['Approach'],
                            y=version_data[metric],
                            name=f'{version}',
                            marker_color=colors[version],
                            opacity=0.7,
                            showlegend=False
                        ),
                        row=2, col=i+1
                    )
        
        fig.update_layout(
            title="Trace Experiments: Detailed vs Undefined Prompts",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_summary_comparison(self):
        """Create an overall summary comparison across all experiments"""
        summary_data = []
        
        for exp_name, df in self.experiments.items():
            if df is not None:
                for version in ['Detailed', 'Undefined']:
                    version_data = df[df['Prompt_Version'] == version]
                    
                    # Extract common metrics where available
                    if 'Accuracy' in df.columns:
                        avg_accuracy = version_data['Accuracy'].mean()
                    elif 'Accuracy_GT_to_Pred (%)' in df.columns:
                        avg_accuracy = version_data['Accuracy_GT_to_Pred (%)'].mean()
                    else:
                        avg_accuracy = None
                    
                    # Similar for other metrics
                    avg_precision = version_data['Precision'].mean() if 'Precision' in df.columns else (
                        version_data['Precision (%)'].mean() if 'Precision (%)' in df.columns else None)
                    
                    avg_f1 = version_data['F1'].mean() if 'F1' in df.columns else (
                        version_data['F1 (%)'].mean() if 'F1 (%)' in df.columns else (
                            version_data['F1_Score'].mean() if 'F1_Score' in df.columns else None))
                    
                    summary_data.append({
                        'Experiment': exp_name,
                        'Version': version,
                        'Avg_Accuracy': avg_accuracy,
                        'Avg_Precision': avg_precision,
                        'Avg_F1': avg_f1,
                        'Sample_Count': len(version_data)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Average Accuracy', 'Average Precision', 'Average F1']
        )
        
        colors = {'Detailed': '#4169E1', 'Undefined': '#FF1493'}
        
        for i, metric in enumerate(['Avg_Accuracy', 'Avg_Precision', 'Avg_F1']):
            for version in ['Detailed', 'Undefined']:
                version_data = summary_df[summary_df['Version'] == version]
                valid_data = version_data.dropna(subset=[metric])
                
                fig.add_trace(
                    go.Bar(
                        x=valid_data['Experiment'],
                        y=valid_data[metric],
                        name=f'{version}',
                        marker_color=colors[version],
                        opacity=0.7,
                        showlegend=(i == 0)
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title="Summary Comparison Across All Experiments",
            height=500,
            showlegend=True
        )
        
        return fig

# Initialize the comparator
comparator = ExperimentComparator()

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Experiment Results Comparison Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.H3("Data Loading Instructions"),
        html.P("Use the load_data method to load your CSV files:"),
        html.Pre("""
# Example usage:
comparator.load_data('nl2pl', 'path/to/detailed_nl2pl.csv', 'path/to/undefined_nl2pl.csv')
comparator.load_data('nl2futureltl', 'path/to/detailed_nl2futureltl.csv', 'path/to/undefined_nl2futureltl.csv')
# ... repeat for all experiments

# Then run the app:
app.run_server(debug=True)
        """, style={'backgroundColor': '#f0f0f0', 'padding': '10px'})
    ], style={'margin': '20px'}),
    
    html.Div(id='plots-container')
])

@app.callback(
    Output('plots-container', 'children'),
    Input('plots-container', 'id')
)
def update_plots(_):
    plots = []
    
    # NL2PL
    nl2pl_fig = comparator.create_nl2pl_comparison()
    if nl2pl_fig:
        plots.append(dcc.Graph(figure=nl2pl_fig))
    
    # LTL experiments
    ltl_fig = comparator.create_ltl_comparison(['nl2futureltl', 'nl2pastltl', 'textbook_nl2futureltl'])
    if ltl_fig:
        plots.append(dcc.Graph(figure=ltl_fig))
    
    # WFF
    wff_fig = comparator.create_wff_comparison()
    if wff_fig:
        plots.append(dcc.Graph(figure=wff_fig))
    
    # Trace experiments
    trace_fig = comparator.create_trace_experiments_comparison()
    if trace_fig:
        plots.append(dcc.Graph(figure=trace_fig))
    
    # Summary
    summary_fig = comparator.create_summary_comparison()
    if summary_fig:
        plots.append(dcc.Graph(figure=summary_fig))
    
    return plots

# Usage example and utility functions
def load_all_experiments(base_path="./"):
    """
    Utility function to load all experiments at once
    Assumes files are named like: detailed_nl2pl.csv, undefined_nl2pl.csv, etc.
    """
    experiments = ['nl2pl', 'nl2futureltl', 'nl2pastltl', 'textbook_nl2futureltl', 
                  'trace_characterization', 'trace_generation', 'wff']
    
    for exp in experiments:
        detailed_path = f"{base_path}/detailed_{exp}.csv"
        undefined_path = f"{base_path}/undefined_{exp}.csv"
        
        if Path(detailed_path).exists() and Path(undefined_path).exists():
            comparator.load_data(exp, detailed_path, undefined_path)
        else:
            print(f"Files not found for {exp}: {detailed_path}, {undefined_path}")

def export_comparison_images(output_dir="./comparison_plots/"):
    """Export all comparison plots as images"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {
        'nl2pl': comparator.create_nl2pl_comparison(),
        'ltl_experiments': comparator.create_ltl_comparison(['nl2futureltl', 'nl2pastltl', 'textbook_nl2futureltl']),
        'wff': comparator.create_wff_comparison(),
        'trace_experiments': comparator.create_trace_experiments_comparison(),
        'summary': comparator.create_summary_comparison()
    }
    
    for name, fig in figures.items():
        if fig is not None:
            fig.write_image(f"{output_dir}/{name}_comparison.png", width=1200, height=800)
            fig.write_html(f"{output_dir}/{name}_comparison.html")
            print(f"Exported {name} comparison")

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    defined_nl2futureltl_path = os.path.join(script_dir, 'data', 'defined', 'comprehensive_table_future_little_tricky.csv')
    defined_nl2pl_path = os.path.join(script_dir, 'data', 'defined', 'nl2pl_aggregated_results.csv')
    defined_wff_path = os.path.join(script_dir, 'data', 'defined', 'wff_aggregate_metrics.csv')

    # For undefined data:
    undefined_nl2futureltl_path = os.path.join(script_dir, 'data', 'undefined', 'comprehensive_table_future_little_tricky.csv')
    undefined_nl2pl_path = os.path.join(script_dir, 'data', 'undefined', 'nl2pl_aggregated_results.csv')
    undefined_wff_path = os.path.join(script_dir, 'data', 'undefined', 'wff_aggregate_metrics.csv')

    # Load defined and undefined data
    try:
        comparator.load_data('nl2futureltl',
                             defined_nl2futureltl_path,
                             undefined_nl2futureltl_path)
        print(f"Loaded nl2futureltl from: {defined_nl2futureltl_path} and {undefined_nl2futureltl_path}")
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
    export_comparison_images()
    
    # Run the dashboard
    app.run(debug=True)
