import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Experiment Comparison Dashboard",
    page_icon="📊",
    layout="wide"
)

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
        'trace_characterization': {
            'defined': os.path.join(script_dir, 'data', 'defined', 'trace_characterization.csv'),
            'undefined': os.path.join(script_dir, 'data', 'undefined', 'trace_characterization.csv')
        },
        'trace_generation': {
            'defined': os.path.join(script_dir, 'data', 'defined', 'trace_generation.csv'),
            'undefined': os.path.join(script_dir, 'data', 'undefined', 'trace_generation.csv')
        }
    }
    
    # Load data for each experiment
    for exp_name, paths in experiments_paths.items():
        try:
            comparator.load_data(exp_name, paths['defined'], paths['undefined'])
            st.success(f"✅ Loaded {exp_name}")
        except FileNotFoundError as e:
            st.error(f"❌ Error loading {exp_name}: {e}")
    
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
        opacity=0.8
    ))
    
    # Add bars for undefined prompts
    fig.add_trace(go.Bar(
        name='Not-Well-Defined Prompt',
        x=x_labels,
        y=undefined_values,
        marker_color='#e74c3c',
        text=[f'{val:.2f}' for val in undefined_values],
        textposition='auto',
        opacity=0.8
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

def create_side_by_side_comparison(data, experiment_name, selected_metrics, metric_mappings, comparison_type="prompt_type"):
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
    st.title("🔬 Experiment Comparison Dashboard")
    st.markdown("Compare performance between **Defined** and **Not-Well-Defined** prompts across different experiments")
    
    # Load data
    with st.spinner("Loading experiment data..."):
        comparator = load_all_data()
    
    metric_mappings = get_metric_mappings()
    
    # Sidebar for controls
    st.sidebar.header("📊 Controls")
    
    # Comparison type selection
    comparison_type = st.sidebar.radio(
        "Comparison Type",
        ["Prompt Type", "Model-Prompt", "Model-Approach Analysis"],
        help="Choose between comparing prompt types, model-prompt combinations, or detailed model-approach analysis"
    )
    
    # Experiment selection
    available_experiments = list(comparator.data.keys())
    selected_experiment = st.sidebar.selectbox(
        "Select Experiment",
        available_experiments,
        index=0 if available_experiments else 0
    )
    
    if selected_experiment and selected_experiment in metric_mappings:
        available_metrics = list(metric_mappings[selected_experiment].keys())
        
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
        
        # Show available models for Model-Prompt comparison
        elif comparison_type == "Model-Prompt":
            st.sidebar.subheader("Available Models")
            experiment_data = comparator.data[selected_experiment]
            defined_df = experiment_data['defined']
            undefined_df = experiment_data['undefined']
            
            model_col = None
            for col in defined_df.columns:
                if 'model' in col.lower():
                    model_col = col
                    break
            
            if model_col:
                defined_models = list(defined_df[model_col].unique())
                undefined_models = list(undefined_df[model_col].unique())
                all_models = list(set(defined_models + undefined_models))
                
                st.sidebar.write("**Models in data:**")
                for model in all_models:
                    defined_count = len(defined_df[defined_df[model_col] == model])
                    undefined_count = len(undefined_df[undefined_df[model_col] == model])
                    st.sidebar.write(f"• {model}: {defined_count} defined, {undefined_count} undefined")
            else:
                st.sidebar.warning("No model column found in the data")
        
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
                st.subheader("📋 Detailed Statistics")
                stats_df = create_model_approach_statistics(
                    comparator,
                    selected_experiment,
                    selected_metric,
                    metric_mappings
                )
                
                if stats_df is not None:
                    st.dataframe(stats_df, use_container_width=True)
                
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
                    comparison_df = create_side_by_side_comparison(
                        comparator,
                        selected_experiment,
                        selected_metrics,
                        metric_mappings,
                        comparison_mode
                    )
                    
                    if comparison_df is not None:
                        st.dataframe(comparison_df, use_container_width=True)
            
            # Additional analysis section
            st.subheader("🔍 Additional Analysis")
            
            # Show data distribution for relevant comparison types
            if comparison_type in ["Model-Prompt", "Model-Approach Analysis"]:
                st.subheader("📊 Data Distribution")
                experiment_data = comparator.data[selected_experiment]
                defined_df = experiment_data['defined']
                undefined_df = experiment_data['undefined']
                
                model_col = None
                for col in defined_df.columns:
                    if 'model' in col.lower():
                        model_col = col
                        break
                
                if model_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Defined Prompts by Model**")
                        defined_counts = defined_df[model_col].value_counts()
                        st.bar_chart(defined_counts)
                    
                    with col2:
                        st.write("**Undefined Prompts by Model**")
                        undefined_counts = undefined_df[model_col].value_counts()
                        st.bar_chart(undefined_counts)
                    
                    # Show approach distribution if available
                    if comparison_type == "Model-Approach Analysis":
                        approach_col = None
                        for col in defined_df.columns:
                            col_lower = col.lower()
                            if any(keyword in col_lower for keyword in ['approach', 'method', 'shot', 'prompt']):
                                approach_col = col
                                break
                        
                        if approach_col:
                            st.subheader("📊 Approach Distribution")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Defined Prompts by Approach**")
                                defined_approach_counts = defined_df[approach_col].value_counts()
                                st.bar_chart(defined_approach_counts)
                            
                            with col2:
                                st.write("**Undefined Prompts by Approach**")
                                undefined_approach_counts = undefined_df[approach_col].value_counts()
                                st.bar_chart(undefined_approach_counts)
            
            # Show raw data if requested
            if st.checkbox("Show Raw Data"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Defined Prompt Data**")
                    st.dataframe(comparator.data[selected_experiment]['defined'])
                
                with col2:
                    st.write("**Not-Well-Defined Prompt Data**")
                    st.dataframe(comparator.data[selected_experiment]['undefined'])
        
        else:
            st.warning("Please select at least one metric to display the comparison.")
    
    else:
        st.error("No experiments available or selected experiment not found.")

if __name__ == "__main__":
    main()