import streamlit as st

# Set page config - MUST be the first Streamlit command

# Configure page
st.set_page_config(
    page_title="Experiment Results Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .comparison-section {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class ExperimentComparator:
    def __init__(self):
        self.data = {}
        
    def load_data(self, experiment_name, defined_path, undefined_path):
        """Load data for both defined and undefined experiments"""
        defined_df = pd.read_csv(defined_path)
        undefined_df = pd.read_csv(undefined_path)
        
        self.data[experiment_name] = {
            'defined': defined_df,
            'undefined': undefined_df
        }

@st.cache_data
def load_all_data():
    """Load all experiment data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    comparator = ExperimentComparator()
    
    # Define file paths
    experiments_paths = {
        'nl2futureltl': {
            'defined': os.path.join(script_dir, 'data', 'defined', 'comprehensive_table_future_little_tricky.csv'),
            'undefined': os.path.join(script_dir, 'data', 'undefined', 'comprehensive_table_future_little_tricky.csv')
        },
        'nl2pl': {
            'defined': os.path.join(script_dir, 'data', 'defined', 'nl2pl_aggregated_results.csv'),
            'undefined': os.path.join(script_dir, 'data', 'undefined', 'nl2pl_aggregated_results.csv')
        },
        'wff': {
            'defined': os.path.join(script_dir, 'data', 'defined', 'wff_aggregate_metrics.csv'),
            'undefined': os.path.join(script_dir, 'data', 'undefined', 'wff_aggregate_metrics.csv')
        },
        'textbook_nl2futureltl': {
            'defined': os.path.join(script_dir, 'data', 'defined', 'comprehensive_table_future_textbook.csv'),
            'undefined': os.path.join(script_dir, 'data', 'undefined', 'comprehensive_table_future_textbook.csv')
        },
    #     'trace_characterization': {
    #         'defined': os.path.join(script_dir, 'data', 'defined', 'trace_characterization.csv'),
    #         'undefined': os.path.join(script_dir, 'data', 'undefined', 'trace_characterization.csv')
    #     },
    #     'trace_generation': {
    #         'defined': os.path.join(script_dir, 'data', 'defined', 'trace_generation.csv'),
    #         'undefined': os.path.join(script_dir, 'data', 'undefined', 'trace_generation.csv')
    #     }
    }
    

    # return comparator
    for exp_name, paths in experiments_paths.items():
        comparator.load_data(exp_name, paths['defined'], paths['undefined'])
    
    return comparator

def get_metric_mappings():
    """Define metric mappings for each experiment"""
    return {
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
        },
        'textbook_nl2futureltl': {
            'GT→Pred Accuracy': 'Accuracy_GT_to_Pred (%)',
            'Pred→GT Accuracy': 'Accuracy_Pred_to_GT (%)',
            'Equivalence Accuracy': 'Equivalence_Accuracy (%)',
            'Syntactic Correctness': 'Syntactic_Correctness_Rate (%)',
            'Syntactic Match': 'Syntactic_Match_Rate (%)',
            'Precision': 'Precision (%)',
            'Recall': 'Recall (%)',
            'F1 Score': 'F1 (%)'
        }
    }

def create_single_metric_comparison(data, experiment_name, selected_metric, metric_mappings):
    """Create a comparison chart for a single metric between defined and undefined"""
    
    if experiment_name not in data.data:
        st.error(f"No data available for {experiment_name}")
        return None
    
    experiment_data = data.data[experiment_name]
    defined_df = experiment_data['defined']
    undefined_df = experiment_data['undefined']
    
    # Get the actual column name from the mapping
    if selected_metric not in metric_mappings[experiment_name]:
        st.error(f"Metric {selected_metric} not found in mappings")
        return None
        
    actual_col = metric_mappings[experiment_name][selected_metric]
    
    if actual_col not in defined_df.columns or actual_col not in undefined_df.columns:
        st.error(f"Column {actual_col} not found in data")
        return None
    
    # Calculate statistics
    defined_mean = defined_df[actual_col].mean()
    undefined_mean = undefined_df[actual_col].mean()
    defined_std = defined_df[actual_col].std()
    undefined_std = undefined_df[actual_col].std()
    
    # Create the comparison chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        name='Defined Prompt',
        x=['Defined'],
        y=[defined_mean],
        marker_color='#3498db',
        text=f'{defined_mean:.3f}',
        textposition='auto',
        error_y=dict(type='data', array=[defined_std], visible=True)
    ))
    
    fig.add_trace(go.Bar(
        name='Not-Well-Defined Prompt',
        x=['Not-Well-Defined'],
        y=[undefined_mean],
        marker_color='#e74c3c',
        text=f'{undefined_mean:.3f}',
        textposition='auto',
        error_y=dict(type='data', array=[undefined_std], visible=True)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{experiment_name.upper()} - {selected_metric}',
        xaxis_title='Prompt Type',
        yaxis_title=f'{selected_metric} Score',
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig, {
        'defined_mean': defined_mean,
        'undefined_mean': undefined_mean,
        'defined_std': defined_std,
        'undefined_std': undefined_std,
        'difference': defined_mean - undefined_mean,
        'defined_count': len(defined_df),
        'undefined_count': len(undefined_df)
    }

def create_detailed_comparison_chart(comparator, experiment, metric, models=None, approaches=None):
    """Create a detailed comparison chart showing models and approaches"""
    
    metric_col = comparator.metric_mappings[experiment].get(metric, metric)
    
    detailed_data = []
    
    for prompt_type in ['defined', 'undefined']:
        data = comparator.experiments[experiment][prompt_type]
        if data is not None and metric_col in data.columns:
            # Find model and approach columns
            model_col = None
            approach_col = None
            
            for col in data.columns:
                if 'model' in col.lower() or col == 'Model':
                    model_col = col
                if 'approach' in col.lower() or col == 'Approach':
                    approach_col = col
            
            # Process data
            for _, row in data.iterrows():
                if pd.notna(row[metric_col]):
                    model = row[model_col] if model_col else 'Unknown'
                    approach = row[approach_col] if approach_col else 'Unknown'
                    
                    # Filter if specified
                    if models and model not in models:
                        continue
                    if approaches and approach not in approaches:
                        continue
                    
                    detailed_data.append({
                        'Prompt Type': prompt_type.title(),
                        'Model': model,
                        'Approach': approach,
                        'Value': row[metric_col],
                        'Category': f"{model} - {approach}"
                    })
    
    if not detailed_data:
        st.warning("No detailed data available for the selected combination.")
        return None
    
    df_detailed = pd.DataFrame(detailed_data)
    
    # Create grouped bar chart
    fig = px.bar(
        df_detailed,
        x='Category',
        y='Value',
        color='Prompt Type',
        title=f"Detailed Comparison: {experiment.replace('_', ' ').title()} - {metric}",
        labels={'Value': metric, 'Category': 'Model - Approach'},
        color_discrete_map={'Defined': '#667eea', 'Undefined': '#764ba2'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        template="plotly_white"
    )
    
    return fig
def create_detailed_statistics_single_metric(stats, selected_metric):
    """Create detailed statistics display for single metric"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Defined Prompt",
            value=f"{stats['defined_mean']:.3f}",
            delta=f"±{stats['defined_std']:.3f}"
        )
        st.caption(f"Count: {stats['defined_count']}")
    
    with col2:
        st.metric(
            label="Not-Well-Defined Prompt", 
            value=f"{stats['undefined_mean']:.3f}",
            delta=f"±{stats['undefined_std']:.3f}"
        )
        st.caption(f"Count: {stats['undefined_count']}")
    
    with col3:
        delta_color = "normal" if stats['difference'] > 0 else "inverse"
        st.metric(
            label="Difference",
            value=f"{stats['difference']:.3f}",
            delta=f"{'Higher' if stats['difference'] > 0 else 'Lower'} for Defined",
            delta_color=delta_color
        )
        
        # Performance indicator
        if abs(stats['difference']) > 0.1:
            st.caption("🔍 **Significant difference**")
        elif abs(stats['difference']) > 0.05:
            st.caption("⚠️ **Moderate difference**")
        else:
            st.caption("✅ **Small difference**")

def create_distribution_comparison(data, experiment_name, selected_metric, metric_mappings):
    """Create distribution comparison for the selected metric"""
    
    experiment_data = data.data[experiment_name]
    defined_df = experiment_data['defined']
    undefined_df = experiment_data['undefined']
    
    actual_col = metric_mappings[experiment_name][selected_metric]
    
    # Create distribution plot
    fig = go.Figure()
    
    # Add histograms
    fig.add_trace(go.Histogram(
        x=defined_df[actual_col],
        name='Defined Prompt',
        opacity=0.5,
        marker_color='#3498db',
        nbinsx=20
    ))
    
    fig.add_trace(go.Histogram(
        x=undefined_df[actual_col],
        name='Not-Well-Defined Prompt',
        opacity=0.5,
        marker_color='#e74c3c',
        nbinsx=20
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Distribution of {selected_metric} Scores',
        xaxis_title=f'{selected_metric} Score',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        showlegend=True
    )
    
    return fig

def create_side_by_side_comparison(data, experiment_name, selected_metric, metric_mappings):
    """Create side-by-side comparison table for single metric"""
    
    if experiment_name not in data.data:
        return None
    
    experiment_data = data.data[experiment_name]
    defined_df = experiment_data['defined']
    undefined_df = experiment_data['undefined']
    
    actual_col = metric_mappings[experiment_name][selected_metric]
    
    if actual_col not in defined_df.columns or actual_col not in undefined_df.columns:
        return None
    
    # Calculate statistics
    defined_stats = {
        'mean': defined_df[actual_col].mean(),
        'std': defined_df[actual_col].std(),
        'min': defined_df[actual_col].min(),
        'max': defined_df[actual_col].max(),
        'median': defined_df[actual_col].median(),
        'count': len(defined_df)
    }
    
    undefined_stats = {
        'mean': undefined_df[actual_col].mean(),
        'std': undefined_df[actual_col].std(),
        'min': undefined_df[actual_col].min(),
        'max': undefined_df[actual_col].max(),
        'median': undefined_df[actual_col].median(),
        'count': len(undefined_df)
    }
    
    # Create comparison table
    comparison_data = []
    for stat_name in ['mean', 'std', 'min', 'max', 'median', 'count']:
        comparison_data.append({
            'Statistic': stat_name.capitalize(),
            'Defined Prompt': f'{defined_stats[stat_name]:.3f}' if stat_name != 'count' else f'{defined_stats[stat_name]}',
            'Not-Well-Defined Prompt': f'{undefined_stats[stat_name]:.3f}' if stat_name != 'count' else f'{undefined_stats[stat_name]}',
            'Difference': f'{defined_stats[stat_name] - undefined_stats[stat_name]:.3f}' if stat_name != 'count' else f'{defined_stats[stat_name] - undefined_stats[stat_name]}'
        })
    
    return pd.DataFrame(comparison_data)

def create_model_approach_comparison(data, experiment_name, selected_metric, metric_mappings):
    """Create a grouped bar chart comparing models and approaches for a specific metric"""
    
    if experiment_name not in data.data:
        st.error(f"No data available for {experiment_name}")
        return None
    
    if selected_metric not in metric_mappings[experiment_name]:
        st.error(f"Metric {selected_metric} not found in mappings")
        return None
    
    actual_metric = metric_mappings[experiment_name][selected_metric]
    experiment_data = data.data[experiment_name]
    defined_df = experiment_data['defined']
    undefined_df = experiment_data['undefined']
    
    # Check if the metric exists in both dataframes
    if actual_metric not in defined_df.columns or actual_metric not in undefined_df.columns:
        st.error(f"Metric {actual_metric} not found in data columns")
        return None
    
    # Find model and approach columns
    model_col = None
    approach_col = None
    
    for col in defined_df.columns:
        col_lower = col.lower()
        if 'model' in col_lower:
            model_col = col
        elif any(keyword in col_lower for keyword in ['approach', 'method', 'shot', 'prompt']):
            approach_col = col
    
    if not model_col:
        st.warning("No model column found in data. Looking for columns with 'model' in the name.")
        return None
    
    # Get unique models and approaches
    defined_models = defined_df[model_col].unique()
    undefined_models = undefined_df[model_col].unique()
    all_models = sorted(list(set(list(defined_models) + list(undefined_models))))
    
    if approach_col:
        defined_approaches = defined_df[approach_col].unique()
        undefined_approaches = undefined_df[approach_col].unique()
        all_approaches = sorted(list(set(list(defined_approaches) + list(undefined_approaches))))
    else:
        all_approaches = ['All Data']
    
    # Create the figure
    fig = go.Figure()
    
    # Color schemes
    colors_defined = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    colors_undefined = ['#2980b9', '#27ae60', '#e67e22', '#c0392b', '#8e44ad', '#16a085', '#d35400', '#2c3e50']
    
    # Create x-axis labels (combinations of model and approach)
    x_labels = []
    defined_values = []
    undefined_values = []
    
    for model in all_models:
        for approach in all_approaches:
            if approach_col:
                # Filter by both model and approach
                defined_subset = defined_df[(defined_df[model_col] == model) & (defined_df[approach_col] == approach)]
                undefined_subset = undefined_df[(undefined_df[model_col] == model) & (undefined_df[approach_col] == approach)]
                x_label = f"{model}\n{approach}"
            else:
                # Filter by model only
                defined_subset = defined_df[defined_df[model_col] == model]
                undefined_subset = undefined_df[undefined_df[model_col] == model]
                x_label = model
            
            if len(defined_subset) > 0 and len(undefined_subset) > 0:
                defined_mean = defined_subset[actual_metric].mean()
                undefined_mean = undefined_subset[actual_metric].mean()
                
                x_labels.append(x_label)
                defined_values.append(defined_mean)
                undefined_values.append(undefined_mean)
    
    if not x_labels:
        st.warning("No matching model-approach combinations found in both defined and undefined data")
        return None
    
    # Add bars for defined prompts
    fig.add_trace(go.Bar(
        name='Defined Prompt',
        x=x_labels,
        y=defined_values,
        marker_color='#3498db',
        text=[f'{val:.2f}' for val in defined_values],
        textposition='auto',
        # opacity=0.1
    ))
    
    # Add bars for undefined prompts
    fig.add_trace(go.Bar(
        name='Not-Well-Defined Prompt',
        x=x_labels,
        y=undefined_values,
        marker_color='#e74c3c',
        text=[f'{val:.2f}' for val in undefined_values],
        textposition='auto',
        # opacity=0.8
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{experiment_name.upper()} - {selected_metric} Comparison by Model and Approach',
        xaxis_title='Model - Approach',
        yaxis_title=selected_metric,
        barmode='group',
        height=600,
        showlegend=True,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_model_approach_statistics(data, experiment_name, selected_metric, metric_mappings):
    """Create detailed statistics table for model-approach combinations"""
    
    if experiment_name not in data.data or selected_metric not in metric_mappings[experiment_name]:
        return None
    
    actual_metric = metric_mappings[experiment_name][selected_metric]
    experiment_data = data.data[experiment_name]
    defined_df = experiment_data['defined']
    undefined_df = experiment_data['undefined']
    
    # Find model and approach columns
    model_col = None
    approach_col = None
    
    for col in defined_df.columns:
        col_lower = col.lower()
        if 'model' in col_lower:
            model_col = col
        elif any(keyword in col_lower for keyword in ['approach', 'method', 'shot', 'prompt']):
            approach_col = col
    
    if not model_col:
        return None
    
    # Get unique models and approaches
    defined_models = defined_df[model_col].unique()
    undefined_models = undefined_df[model_col].unique()
    all_models = sorted(list(set(list(defined_models) + list(undefined_models))))
    
    if approach_col:
        defined_approaches = defined_df[approach_col].unique()
        undefined_approaches = undefined_df[approach_col].unique()
        all_approaches = sorted(list(set(list(defined_approaches) + list(undefined_approaches))))
    else:
        all_approaches = ['All Data']
    
    # Create statistics table
    stats_data = []
    
    for model in all_models:
        for approach in all_approaches:
            if approach_col:
                defined_subset = defined_df[(defined_df[model_col] == model) & (defined_df[approach_col] == approach)]
                undefined_subset = undefined_df[(undefined_df[model_col] == model) & (undefined_df[approach_col] == approach)]
            else:
                defined_subset = defined_df[defined_df[model_col] == model]
                undefined_subset = undefined_df[undefined_df[model_col] == model]
            
            if len(defined_subset) > 0 and len(undefined_subset) > 0:
                defined_mean = defined_subset[actual_metric].mean()
                undefined_mean = undefined_subset[actual_metric].mean()
                defined_std = defined_subset[actual_metric].std()
                undefined_std = undefined_subset[actual_metric].std()
                
                stats_data.append({
                    'Model': model,
                    'Approach': approach if approach_col else 'N/A',
                    'Defined Mean': f'{defined_mean:.3f}',
                    'Defined Std': f'{defined_std:.3f}',
                    'Undefined Mean': f'{undefined_mean:.3f}',
                    'Undefined Std': f'{undefined_std:.3f}',
                    'Difference': f'{defined_mean - undefined_mean:.3f}',
                    'Defined Count': len(defined_subset),
                    'Undefined Count': len(undefined_subset)
                })
    
    if stats_data:
        return pd.DataFrame(stats_data)
    return None

def create_grouped_bar_chart(data, experiment_name, selected_metrics, metric_mappings, comparison_type="prompt_type"):
    """Create a grouped bar chart with different comparison options"""
    
    if experiment_name not in data.data:
        st.error(f"No data available for {experiment_name}")
        return None
    
    experiment_data = data.data[experiment_name]
    defined_df = experiment_data['defined']
    undefined_df = experiment_data['undefined']
    
    # Get the actual column names from the mapping
    actual_metrics = []
    display_metrics = []
    
    for display_name in selected_metrics:
        if display_name in metric_mappings[experiment_name]:
            actual_col = metric_mappings[experiment_name][display_name]
            if actual_col in defined_df.columns and actual_col in undefined_df.columns:
                actual_metrics.append(actual_col)
                display_metrics.append(display_name)
    
    if not actual_metrics:
        st.warning("No valid metrics selected or available in the data")
        return None
    
    # Create the grouped bar chart
    fig = go.Figure()
    
    if comparison_type == "prompt_type":
        # Original comparison: defined vs undefined
        defined_means = []
        undefined_means = []
        
        for metric in actual_metrics:
            defined_mean = defined_df[metric].mean()
            undefined_mean = undefined_df[metric].mean()
            defined_means.append(defined_mean)
            undefined_means.append(undefined_mean)
        
        fig.add_trace(go.Bar(
            name='Defined Prompt',
            x=display_metrics,
            y=defined_means,
            marker_color='#3498db',
            text=[f'{val:.2f}' for val in defined_means],
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='Not-Well-Defined Prompt',
            x=display_metrics,
            y=undefined_means,
            marker_color='#e74c3c',
            text=[f'{val:.2f}' for val in undefined_means],
            textposition='auto',
        ))
        
        title = f'{experiment_name.upper()} - Comparison of Defined vs Not-Well-Defined Prompts'
    
    elif comparison_type == "model_prompt":
        # Model-Prompt combination comparison
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        color_idx = 0
        
        # Combine both dataframes with labels
        combined_data = []
        
        # Process defined data
        model_col = None
        prompt_col = None
        
        for col in defined_df.columns:
            if 'model' in col.lower():
                model_col = col
                break
        
        if model_col:
            for model in defined_df[model_col].unique():
                model_data = defined_df[defined_df[model_col] == model]
                means = []
                
                for metric in actual_metrics:
                    means.append(model_data[metric].mean())
                
                fig.add_trace(go.Bar(
                    name=f'{model} (Defined)',
                    x=display_metrics,
                    y=means,
                    marker_color=colors[color_idx % len(colors)],
                    text=[f'{val:.2f}' for val in means],
                    textposition='auto',
                ))
                color_idx += 1
            
            # Process undefined data
            for model in undefined_df[model_col].unique():
                model_data = undefined_df[undefined_df[model_col] == model]
                means = []
                
                for metric in actual_metrics:
                    means.append(model_data[metric].mean())
                
                fig.add_trace(go.Bar(
                    name=f'{model} (Undefined)',
                    x=display_metrics,
                    y=means,
                    marker_color=colors[color_idx % len(colors)],
                    text=[f'{val:.2f}' for val in means],
                    textposition='auto',
                ))
                color_idx += 1
        
        else:
            # Fallback to prompt type comparison if no model column
            st.warning("No model column found in data. Falling back to prompt type comparison.")
            return create_grouped_bar_chart(data, experiment_name, selected_metrics, metric_mappings, "prompt_type")
        
        title = f'{experiment_name.upper()} - Model Comparison Across Prompt Types'
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Metrics',
        yaxis_title='Score',
        barmode='group',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_detailed_side_by_side_comparison(data, experiment_name, selected_metrics, metric_mappings, comparison_type="prompt_type"):
    """Create side-by-side comparison with detailed statistics"""
    
    if experiment_name not in data.data:
        return None
    
    experiment_data = data.data[experiment_name]
    defined_df = experiment_data['defined']
    undefined_df = experiment_data['undefined']
    
    # Create comparison table
    comparison_data = []
    
    if comparison_type == "prompt_type":
        # Original prompt type comparison
        for display_name in selected_metrics:
            if display_name in metric_mappings[experiment_name]:
                actual_col = metric_mappings[experiment_name][display_name]
                if actual_col in defined_df.columns and actual_col in undefined_df.columns:
                    defined_mean = defined_df[actual_col].mean()
                    undefined_mean = undefined_df[actual_col].mean()
                    defined_std = defined_df[actual_col].std()
                    undefined_std = undefined_df[actual_col].std()
                    
                    comparison_data.append({
                        'Metric': display_name,
                        'Defined Mean': f'{defined_mean:.3f}',
                        'Defined Std': f'{defined_std:.3f}',
                        'Undefined Mean': f'{undefined_mean:.3f}',
                        'Undefined Std': f'{undefined_std:.3f}',
                        'Difference': f'{defined_mean - undefined_mean:.3f}'
                    })
    
    elif comparison_type == "model_prompt":
        # Model-prompt comparison
        model_col = None
        for col in defined_df.columns:
            if 'model' in col.lower():
                model_col = col
                break
        
        if model_col:
            all_models = list(defined_df[model_col].unique()) + list(undefined_df[model_col].unique())
            unique_models = list(set(all_models))
            
            for model in unique_models:
                defined_model_data = defined_df[defined_df[model_col] == model]
                undefined_model_data = undefined_df[undefined_df[model_col] == model]
                
                if len(defined_model_data) > 0 and len(undefined_model_data) > 0:
                    for display_name in selected_metrics:
                        if display_name in metric_mappings[experiment_name]:
                            actual_col = metric_mappings[experiment_name][display_name]
                            if actual_col in defined_df.columns and actual_col in undefined_df.columns:
                                defined_mean = defined_model_data[actual_col].mean()
                                undefined_mean = undefined_model_data[actual_col].mean()
                                defined_std = defined_model_data[actual_col].std()
                                undefined_std = undefined_model_data[actual_col].std()
                                
                                comparison_data.append({
                                    'Model': model,
                                    'Metric': display_name,
                                    'Defined Mean': f'{defined_mean:.3f}',
                                    'Defined Std': f'{defined_std:.3f}',
                                    'Undefined Mean': f'{undefined_mean:.3f}',
                                    'Undefined Std': f'{undefined_std:.3f}',
                                    'Difference': f'{defined_mean - undefined_mean:.3f}'
                                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    return None

def main():
    # Header
    st.markdown('<div class="main-header"><h1>🔬 Experiment Results Dashboard</h1><p>Compare Defined vs Undefined Prompts Across Different Experiments</p></div>', unsafe_allow_html=True)

    with st.spinner("Loading experiment data..."):
        comparator = load_all_data()
    
    metric_mappings = get_metric_mappings()

    # # Sidebar for controls
    st.sidebar.header("📊 Controls")
    comparison_type = st.sidebar.radio(
        "Comparison Type",
        ["Model-Approach Analysis", "Prompt Type"],
        help="Choose between comparing prompt types, or detailed model-approach analysis"
    )
    
    # Experiment selection
    available_experiments = list(comparator.data.keys())
    selected_experiment = st.sidebar.selectbox(
        "Select Experiment",
        available_experiments,
        index=0 if available_experiments else 0,
        help="Choose which experiment to analyze"
    )  
    if selected_experiment and selected_experiment in metric_mappings:
        # Metric selection - single select based on selected experiment
        available_metrics = list(metric_mappings[selected_experiment].keys())
        selected_metric = st.sidebar.selectbox(
            "Select Metric",
            available_metrics,
            index=0 if available_metrics else 0,
            help="Choose which metric to compare between defined and undefined prompts"
        )

        if comparison_type == "Model-Approach Analysis":
            # Single metric selection for model-approach analysis
            selected_metric = st.sidebar.selectbox(
                "Select Metric",
                available_metrics,
                help="Select a single metric to analyze across models and approaches"
            )
            selected_metrics = [selected_metric]  # Convert to list for compatibility
        else:
            # Multi-metric selection for other comparison types
            selected_metrics = st.sidebar.multiselect(
                "Select Metrics",
                available_metrics,
                default=available_metrics[:4]  # Default to first 4 metrics
            )
        # Show available models and approaches for Model-Approach Analysis
        if comparison_type == "Model-Approach Analysis":
            st.sidebar.subheader("Data Overview")
            experiment_data = comparator.data[selected_experiment]
            defined_df = experiment_data['defined']
            undefined_df = experiment_data['undefined']
            
            # Find model and approach columns
            model_col = None
            approach_col = None
            
            for col in defined_df.columns:
                col_lower = col.lower()
                if 'model' in col_lower:
                    model_col = col
                elif any(keyword in col_lower for keyword in ['approach', 'method', 'shot', 'prompt']):
                    approach_col = col
            
            if model_col:
                all_models = sorted(list(set(list(defined_df[model_col].unique()) + list(undefined_df[model_col].unique()))))
                st.sidebar.write(f"**Models ({len(all_models)}):**")
                for model in all_models:
                    st.sidebar.write(f"• {model}")
            
            if approach_col:
                all_approaches = sorted(list(set(list(defined_df[approach_col].unique()) + list(undefined_df[approach_col].unique()))))
                st.sidebar.write(f"**Approaches ({len(all_approaches)}):**")
                for approach in all_approaches:
                    st.sidebar.write(f"• {approach}")
            
            # Show column information
            st.sidebar.subheader("Column Information")
            st.sidebar.write(f"**Model Column:** {model_col if model_col else 'Not found'}")
            st.sidebar.write(f"**Approach Column:** {approach_col if approach_col else 'Not found'}")
        
        if selected_metrics:
            # Main content area
            if comparison_type == "Model-Approach Analysis":
                # Single metric analysis with model-approach comparison
                st.subheader(f"🔬 {selected_experiment.upper()} - Model & Approach Analysis")
                st.write(f"**Analyzing:** {selected_metric}")
                
                # Create model-approach comparison chart
                fig = create_model_approach_comparison(
                    comparator,
                    selected_experiment,
                    selected_metric,
                    metric_mappings
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed statistics
                # st.subheader("📋 Detailed Statistics")
                # stats_df = create_model_approach_statistics(
                #     comparator,
                #     selected_experiment,
                #     selected_metric,
                #     metric_mappings
                # )
                
                # if stats_df is not None:
                #     st.dataframe(stats_df, use_container_width=True)
                
            else:
                # Original multi-metric analysis
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    comparison_mode = "prompt_type" if comparison_type == "Prompt Type" else "model_prompt"
                    
                    st.subheader(f"📈 {selected_experiment.upper()} - {comparison_type} Comparison")
                    
                    # Create and display the chart
                    fig = create_grouped_bar_chart(
                        comparator, 
                        selected_experiment, 
                        selected_metrics, 
                        metric_mappings,
                        comparison_mode
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📋 Detailed Statistics")
                    
                    # Create comparison table
                    comparison_df = create_detailed_side_by_side_comparison(
                        comparator,
                        selected_experiment,
                        selected_metrics,
                        metric_mappings,
                        comparison_mode
                    )
                    
                    if comparison_df is not None:
                        st.dataframe(comparison_df, use_container_width=True)
                
        if selected_metric:
            # Main content area
            st.header(f"📈 {selected_experiment.upper()} - {selected_metric}")
            
            # Create the comparison chart and get statistics
            result = create_single_metric_comparison(
                comparator, 
                selected_experiment, 
                selected_metric, 
                metric_mappings
            )
            
            
            if result:
                fig, stats = result
                
                # Display the main chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed statistics
                st.subheader("📊 Detailed Statistics")
                create_detailed_statistics_single_metric(stats, selected_metric)
                
                # Create two columns for additional analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Statistical Summary")
                    comparison_df = create_side_by_side_comparison(
                        comparator,
                        selected_experiment,
                        selected_metric,
                        metric_mappings
                    )
                    
                    if comparison_df is not None:
                        st.dataframe(comparison_df, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Score Distribution")
                    distribution_fig = create_distribution_comparison(
                        comparator,
                        selected_experiment,
                        selected_metric,
                        metric_mappings
                    )
                    st.plotly_chart(distribution_fig, use_container_width=True)
                
                # Additional insights
                st.subheader("🔍 Key Insights")
                
                # Generate insights based on the statistics
                if abs(stats['difference']) > 0.1:
                    if stats['difference'] > 0:
                        st.success(f"✅ **Defined prompts significantly outperform** not-well-defined prompts by {stats['difference']:.3f} points in {selected_metric}")
                    else:
                        st.error(f"❌ **Not-well-defined prompts significantly outperform** defined prompts by {abs(stats['difference']):.3f} points in {selected_metric}")
                elif abs(stats['difference']) > 0.05:
                    st.warning(f"⚠️ **Moderate difference** observed: {'Defined' if stats['difference'] > 0 else 'Not-well-defined'} prompts perform better by {abs(stats['difference']):.3f} points")
                else:
                    st.info(f"ℹ️ **Similar performance** between defined and not-well-defined prompts (difference: {stats['difference']:.3f} points)")
                
                # Show variability insights
                if stats['defined_std'] > stats['undefined_std']:
                    st.info(f"📊 **Defined prompts show higher variability** (std: {stats['defined_std']:.3f}) compared to not-well-defined prompts (std: {stats['undefined_std']:.3f})")
                elif stats['undefined_std'] > stats['defined_std']:
                    st.info(f"📊 **Not-well-defined prompts show higher variability** (std: {stats['undefined_std']:.3f}) compared to defined prompts (std: {stats['defined_std']:.3f})")
                else:
                    st.info(f"📊 **Similar variability** between both prompt types")
                
                # Show raw data if requested
                if st.checkbox("Show Raw Data", help="Display the underlying data used for this comparison"):
                    st.subheader("📄 Raw Data")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Defined Prompt Data**")
                        st.dataframe(comparator.data[selected_experiment]['defined'])
                    
                    with col2:
                        st.write("**Not-Well-Defined Prompt Data**")
                        st.dataframe(comparator.data[selected_experiment]['undefined'])
        
        else:
            st.warning("Please select a metric to display the comparison.")
    
    else:
        st.error("No experiments available or selected experiment not found.")

if __name__ == "__main__":
    main()