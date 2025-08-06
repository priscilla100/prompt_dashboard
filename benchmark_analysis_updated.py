import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from scipy import stats
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Dynamic Benchmark Comparative Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        color: #2E4057;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚀 Dynamic Benchmark Comparative Analysis</h1>', unsafe_allow_html=True)

# Experiment configurations
experiment_options = [
    {'label': 'NL2PL', 'value': 'nl2pl'},
    {'label': 'NL2 Future LTL', 'value': 'nl2futureltl'},
    {'label': 'NL2 Past LTL', 'value': 'nl2pastltl'},
    {'label': 'Textbook NL2 Future LTL', 'value': 'textbook_nl2futureltl'},
    {'label': 'WFF Classification', 'value': 'wff'},
    {'label': 'Trace Generation', 'value': 'trace_generation'},
    {'label': 'Trace Characterization', 'value': 'trace_characterization'}
]

# Analysis modes
ANALYSIS_MODES = {
    'Strategy Comparison': 'strategy',
    'Model-Approach-Metric Analysis': 'model_approach_metric'
}

# Metric mappings
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
    'nl2pastltl': {
        'GT→Pred Accuracy': 'Accuracy_GT_to_Pred (%)',
        'Pred→GT Accuracy': 'Accuracy_Pred_to_GT (%)',
        'Equivalence Accuracy': 'Equivalence_Accuracy (%)',
        'Syntactic Correctness': 'Syntactic_Correctness_Rate (%)',
        'Syntactic Match': 'Syntactic_Match_Rate (%)',
        'Precision': 'Precision (%)',
        'Recall': 'Recall (%)',
        'F1 Score': 'F1 (%)'
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

# File mappings for experiments
file_mappings = {
    'nl2pl': 'nl2pl_aggregated_results.csv',
    'nl2futureltl': 'comprehensive_table_future_little_tricky.csv',
    'nl2pastltl': 'comprehensive_table_past_little_tricky.csv',
    'textbook_nl2futureltl': 'comprehensive_table_future_textbook.csv',
    'wff': 'wff_aggregate_metrics.csv',
    'trace_generation': 'trace_generation.csv',
    'trace_characterization': 'trace_characterization.csv'
}

@st.cache_data
def load_data(strategy, experiment):
    """Load data for specific strategy and experiment"""
    try:
        file_path = f"compare_data/{strategy}/{file_mappings[experiment]}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            st.warning(f"File not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_all_models_from_data(strategies, experiment):
    """Extract all unique models from the datasets"""
    all_models = set()
    for strategy in strategies:
        data = load_data(strategy, experiment)
        if data is not None and 'Model' in data.columns:
            all_models.update(data['Model'].unique())
    return sorted(list(all_models))

def create_model_approach_metric_analysis(strategies, experiment, selected_models, selected_metrics):
    """Create comprehensive model-approach-metric analysis"""
    analysis_data = []
    
    for strategy in strategies:
        data = load_data(strategy, experiment)
        if data is not None and 'Model' in data.columns:
            for model in selected_models:
                model_data = data[data['Model'] == model]
                if not model_data.empty:
                    for metric in selected_metrics:
                        metric_col = metric_mappings[experiment][metric]
                        if metric_col in model_data.columns:
                            values = model_data[metric_col].dropna()
                            if len(values) > 0:
                                analysis_data.append({
                                    'Model': model,
                                    'Approach': strategy,
                                    'Metric': metric,
                                    'Mean': values.mean(),
                                    'Std': values.std(),
                                    'Min': values.min(),
                                    'Max': values.max(),
                                    'Median': values.median(),
                                    'Count': len(values),
                                    'Q25': values.quantile(0.25),
                                    'Q75': values.quantile(0.75)
                                })
    
    return pd.DataFrame(analysis_data)

def create_model_performance_heatmap(analysis_df, metric_focus=None):
    """Create model performance heatmap across approaches"""
    if analysis_df.empty:
        return None
    
    # Filter by metric if specified
    if metric_focus:
        plot_data = analysis_df[analysis_df['Metric'] == metric_focus]
        title_suffix = f" - {metric_focus}"
    else:
        # Use first metric or aggregate
        plot_data = analysis_df.groupby(['Model', 'Approach'])['Mean'].mean().reset_index()
        title_suffix = " - Overall Performance"
    
    if plot_data.empty:
        return None
    
    # Pivot for heatmap
    if 'Mean' not in plot_data.columns:
        plot_data = plot_data.groupby(['Model', 'Approach'])['Mean'].first().reset_index()
    
    heatmap_matrix = plot_data.pivot(index='Model', columns='Approach', values='Mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix.values,
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        colorscale='Viridis',
        text=np.around(heatmap_matrix.values, decimals=3),
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        hoverongaps=False,
        hovertemplate='<b>Model</b>: %{y}<br>' +
                     '<b>Approach</b>: %{x}<br>' +
                     '<b>Performance</b>: %{z:.4f}<extra></extra>',
        colorbar=dict(title="Performance Score")
    ))
    
    fig.update_layout(
        title={
            'text': f'<b>🔥 Model Performance Heatmap{title_suffix}</b>',
            'x': 0.5,
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        xaxis_title="<b>Prompting Approaches</b>",
        yaxis_title="<b>Models</b>",
        height=max(400, len(heatmap_matrix.index) * 40),
        width=800
    )
    
    return fig

def create_model_metric_comparison(analysis_df, selected_models):
    """Create comprehensive model-metric comparison across approaches"""
    if analysis_df.empty:
        return None
    
    fig = make_subplots(
        rows=len(selected_models),
        cols=1,
        subplot_titles=[f"<b>{model}</b>" for model in selected_models],
        vertical_spacing=0.05
    )
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#4ecdc4', '#45b7d1']
    
    for i, model in enumerate(selected_models):
        model_data = analysis_df[analysis_df['Model'] == model]
        
        approaches = model_data['Approach'].unique()
        for j, approach in enumerate(approaches):
            approach_data = model_data[model_data['Approach'] == approach]
            
            fig.add_trace(
                go.Bar(
                    name=f"{approach}" if i == 0 else "",
                    x=approach_data['Metric'],
                    y=approach_data['Mean'],
                    error_y=dict(type='data', array=approach_data['Std']),
                    marker_color=colors[j % len(colors)],
                    text=[f"{val:.3f}" for val in approach_data['Mean']],
                    textposition='auto',
                    showlegend=(i == 0),
                    hovertemplate=f'<b>{approach}</b><br>' +
                                 'Metric: %{x}<br>' +
                                 'Mean: %{y:.4f}<br>' +
                                 'Std: %{error_y.array:.4f}<extra></extra>'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        title={
            'text': '<b>🎯 Model-Metric Performance Across Approaches</b>',
            'x': 0.5,
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        height=300 * len(selected_models),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="<b>Metrics</b>")
    fig.update_yaxes(title_text="<b>Performance Score</b>")
    
    return fig

def create_approach_ranking_analysis(analysis_df):
    """Create approach ranking analysis for each model"""
    if analysis_df.empty:
        return None
    
    ranking_data = []
    
    for model in analysis_df['Model'].unique():
        model_data = analysis_df[analysis_df['Model'] == model]
        
        # Calculate overall performance per approach
        approach_performance = model_data.groupby('Approach')['Mean'].mean().sort_values(ascending=False)
        
        for rank, (approach, score) in enumerate(approach_performance.items(), 1):
            ranking_data.append({
                'Model': model,
                'Approach': approach,
                'Overall_Score': score,
                'Rank': rank,
                'Metric_Count': len(model_data[model_data['Approach'] == approach])
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # Create ranking visualization
    fig = px.bar(
        ranking_df,
        x='Model',
        y='Overall_Score',
        color='Approach',
        text='Rank',
        title='<b>🏆 Approach Rankings by Model</b>',
        labels={
            'Overall_Score': 'Overall Performance Score',
            'Model': 'Models'
        },
        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
    )
    
    fig.update_traces(
        texttemplate='#%{text}',
        textposition='inside',
        textfont_size=18,
        textfont_color='white'
    )
    
    fig.update_layout(
        title={
            'x': 0.5,
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        height=500,
        showlegend=True
    )
    
    return fig, ranking_df

def get_available_strategies(experiment):
    """Get available strategies for a given experiment"""
    available = []
    for strategy in ['Minimal', 'Detailed', 'Python']:
        if experiment in ['nl2pl', 'wff']:
            if strategy != 'Python':  # Python doesn't have these experiments
                available.append(strategy)
        else:
            available.append(strategy)
    return available

def calculate_statistics(data, metric_col):
    """Calculate comprehensive statistics for a metric"""
    if metric_col not in data.columns:
        return {}
    
    values = data[metric_col].dropna()
    if len(values) == 0:
        return {}
    
    stats_dict = {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
        'count': len(values)
    }
    return stats_dict

def create_comparison_dataframe(strategies, experiment, metric):
    """Create a comparison dataframe for selected strategies"""
    comparison_data = []
    
    for strategy in strategies:
        data = load_data(strategy, experiment)
        if data is not None:
            metric_col = metric_mappings[experiment][metric]
            if metric_col in data.columns:
                stats = calculate_statistics(data, metric_col)
                if stats:
                    stats['Strategy'] = strategy
                    stats['Experiment'] = experiment
                    stats['Metric'] = metric
                    comparison_data.append(stats)
    
    return pd.DataFrame(comparison_data)

def create_bar_chart(comparison_df, metric_name):
    """Create an interactive bar chart for strategy comparison"""
    if comparison_df.empty:
        return None
    
    fig = go.Figure()
    
    strategies = comparison_df['Strategy'].unique()
    colors = ['#667eea', '#764ba2', '#f093fb']
    
    for i, strategy in enumerate(strategies):
        strategy_data = comparison_df[comparison_df['Strategy'] == strategy]
        
        fig.add_trace(go.Bar(
            name=strategy,
            x=['Mean', 'Median', 'Max', 'Min'],
            y=[strategy_data['mean'].iloc[0], 
               strategy_data['median'].iloc[0],
               strategy_data['max'].iloc[0],
               strategy_data['min'].iloc[0]],
            marker_color=colors[i % len(colors)],
            text=[f"{val:.3f}" for val in [
                strategy_data['mean'].iloc[0], 
                strategy_data['median'].iloc[0],
                strategy_data['max'].iloc[0],
                strategy_data['min'].iloc[0]
            ]],
            textposition='auto',
            hovertemplate=f'<b>{strategy}</b><br>' +
                         'Statistic: %{x}<br>' +
                         'Value: %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': f'<b>📊 {metric_name} Performance Comparison</b>',
            'x': 0.5,
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        xaxis_title="<b>Statistical Measures</b>",
        yaxis_title=f"<b>{metric_name} Values</b>",
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_heatmap(strategies, experiment, available_metrics):
    """Create a performance heatmap"""
    heatmap_data = []
    
    for strategy in strategies:
        strategy_row = {'Strategy': strategy}
        data = load_data(strategy, experiment)
        
        if data is not None:
            for metric in available_metrics:
                metric_col = metric_mappings[experiment][metric]
                if metric_col in data.columns:
                    mean_val = data[metric_col].dropna().mean()
                    strategy_row[metric] = mean_val
                else:
                    strategy_row[metric] = np.nan
        
        heatmap_data.append(strategy_row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('Strategy')
    
    if heatmap_df.empty:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='Viridis',
        text=np.around(heatmap_df.values, decimals=3),
        texttemplate="%{text}",
        textfont={"size": 18},
        hoverongaps=False,
        hovertemplate='<b>Strategy</b>: %{y}<br>' +
                     '<b>Metric</b>: %{x}<br>' +
                     '<b>Value</b>: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'<b>🔥 Performance Heatmap - {experiment.upper()}</b>',
            'x': 0.5,
            'font': {'size': 24, 'family': 'Arial Black'}
        },
        xaxis_title="<b>Metrics</b>",
        yaxis_title="<b>Strategies</b>",
        height=400
    )
    
    return fig

def create_statistical_summary(comparison_df):
    """Create detailed statistical summary"""
    if comparison_df.empty:
        return None
    
    st.markdown('<div class="section-header">📈 Statistical Analysis & Insights</div>', unsafe_allow_html=True)
    
    # Performance Summary
    cols = st.columns(len(comparison_df))
    for i, row in comparison_df.iterrows():
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{row['Strategy']}</h3>
                <p><strong>Mean:</strong> {row['mean']:.4f}</p>
                <p><strong>Std Dev:</strong> {row['std']:.4f}</p>
                <p><strong>Range:</strong> {row['max']:.4f} - {row['min']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Strategy Comparisons
    st.markdown('<div class="section-header">🔍 Strategy Comparisons</div>', unsafe_allow_html=True)
    
    if len(comparison_df) > 1:
        best_mean = comparison_df.loc[comparison_df['mean'].idxmax()]
        worst_mean = comparison_df.loc[comparison_df['mean'].idxmin()]
        most_consistent = comparison_df.loc[comparison_df['std'].idxmin()]
        
        improvement = ((best_mean['mean'] - worst_mean['mean']) / worst_mean['mean']) * 100
        
        insights = f"""
        <div class="insight-box">
        <h4>🎯 Key Insights:</h4>
        <ul>
            <li><strong>Best Performing Strategy:</strong> {best_mean['Strategy']} (Mean: {best_mean['mean']:.4f})</li>
            <li><strong>Most Consistent Strategy:</strong> {most_consistent['Strategy']} (Std Dev: {most_consistent['std']:.4f})</li>
            <li><strong>Performance Gap:</strong> {improvement:.2f}% difference between best and worst</li>
            <li><strong>Overall Range:</strong> {comparison_df['max'].max():.4f} to {comparison_df['min'].min():.4f}</li>
        </ul>
        </div>
        """
        st.markdown(insights, unsafe_allow_html=True)
        
        # Statistical significance test (if more than 2 strategies)
        if len(comparison_df) >= 2:
            st.markdown('<div class="section-header">📊 Statistical Significance</div>', unsafe_allow_html=True)
            
            # Load raw data for significance testing
            raw_data = {}
            for _, row in comparison_df.iterrows():
                strategy = row['Strategy']
                experiment = row['Experiment']
                metric = row['Metric']
                data = load_data(strategy, experiment)
                if data is not None:
                    metric_col = metric_mappings[experiment][metric]
                    if metric_col in data.columns:
                        raw_data[strategy] = data[metric_col].dropna()
            
            if len(raw_data) >= 2:
                strategies_list = list(raw_data.keys())
                
                # Perform t-tests between strategies
                significance_results = []
                for i in range(len(strategies_list)):
                    for j in range(i+1, len(strategies_list)):
                        strategy1, strategy2 = strategies_list[i], strategies_list[j]
                        if len(raw_data[strategy1]) > 1 and len(raw_data[strategy2]) > 1:
                            t_stat, p_value = stats.ttest_ind(raw_data[strategy1], raw_data[strategy2])
                            significance_results.append({
                                'Comparison': f"{strategy1} vs {strategy2}",
                                'T-Statistic': t_stat,
                                'P-Value': p_value,
                                'Significant': p_value < 0.05
                            })
                
                if significance_results:
                    sig_df = pd.DataFrame(significance_results)
                    st.dataframe(sig_df, use_container_width=True)

# Sidebar for user inputs
st.sidebar.markdown("## 🎛️ Analysis Configuration")

# Analysis Mode Selection
st.sidebar.markdown("### 🔬 Analysis Mode")
analysis_mode = st.sidebar.selectbox(
    "Select Analysis Type:",
    options=list(ANALYSIS_MODES.keys()),
    help="Choose between strategy comparison or model-approach-metric analysis"
)

current_mode = ANALYSIS_MODES[analysis_mode]

if current_mode == 'strategy':
    # Original Strategy Comparison Mode
    st.sidebar.markdown("### 📝 Layer 1: Prompting Strategies")
    all_strategies = ['Minimal', 'Detailed', 'Python']
    selected_strategies = st.sidebar.multiselect(
        "Select Prompting Strategies:",
        options=all_strategies,
        default=['Minimal', 'Detailed'],
        help="Choose one, two, or all three prompting strategies to compare"
    )

    # Layer 2: Experiment Selection
    st.sidebar.markdown("### 🧪 Layer 2: Experiment Selection")
    experiment_labels = [opt['label'] for opt in experiment_options]
    experiment_values = [opt['value'] for opt in experiment_options]

    selected_experiment_label = st.sidebar.selectbox(
        "Select Experiment:",
        options=experiment_labels,
        help="Choose the experiment to analyze"
    )

    selected_experiment = experiment_values[experiment_labels.index(selected_experiment_label)]

    # Filter strategies based on experiment availability
    available_strategies = get_available_strategies(selected_experiment)
    filtered_strategies = [s for s in selected_strategies if s in available_strategies]

    if len(filtered_strategies) != len(selected_strategies):
        unavailable = [s for s in selected_strategies if s not in available_strategies]
        st.sidebar.warning(f"⚠️ {', '.join(unavailable)} not available for {selected_experiment_label}")

    # Layer 3: Metric Selection
    st.sidebar.markdown("### 📏 Layer 3: Metric Selection")
    if selected_experiment in metric_mappings:
        available_metrics = list(metric_mappings[selected_experiment].keys())
        selected_metric = st.sidebar.selectbox(
            "Select Metric:",
            options=available_metrics,
            help="Choose the performance metric to analyze"
        )
    else:
        st.sidebar.error("Invalid experiment selection")
        selected_metric = None

else:
    # Model-Approach-Metric Analysis Mode
    st.sidebar.markdown("### 🧪 Experiment Selection")
    experiment_labels = [opt['label'] for opt in experiment_options]
    experiment_values = [opt['value'] for opt in experiment_options]

    selected_experiment_label = st.sidebar.selectbox(
        "Select Experiment:",
        options=experiment_labels,
        help="Choose the experiment to analyze"
    )

    selected_experiment = experiment_values[experiment_labels.index(selected_experiment_label)]
    
    # Get available strategies for this experiment
    available_strategies = get_available_strategies(selected_experiment)
    
    st.sidebar.markdown("### 📝 Approach Selection")
    selected_approaches = st.sidebar.multiselect(
        "Select Approaches:",
        options=available_strategies,
        default=available_strategies,
        help="Choose which prompting approaches to include"
    )
    
    # Get all models from the data
    if selected_approaches:
        all_models = get_all_models_from_data(selected_approaches, selected_experiment)
        
        st.sidebar.markdown("### 🤖 Model Selection")
        selected_models = st.sidebar.multiselect(
            "Select Models:",
            options=all_models,
            default=all_models[:5] if len(all_models) > 5 else all_models,
            help="Choose which models to analyze"
        )
        
        st.sidebar.markdown("### 📏 Metric Selection")
        if selected_experiment in metric_mappings:
            available_metrics = list(metric_mappings[selected_experiment].keys())
            selected_metrics = st.sidebar.multiselect(
                "Select Metrics:",
                options=available_metrics,
                default=available_metrics[:3] if len(available_metrics) > 3 else available_metrics,
                help="Choose which metrics to analyze"
            )
        else:
            selected_metrics = []
    else:
        selected_models = []
        selected_metrics = []

# Main analysis
if current_mode == 'strategy':
    # Original Strategy Comparison Analysis
    if filtered_strategies and selected_experiment and selected_metric:
        st.markdown(f"## 🎯 Analysis: {selected_experiment_label} - {selected_metric}")
        
        # Create comparison dataframe
        comparison_df = create_comparison_dataframe(filtered_strategies, selected_experiment, selected_metric)
        
        if not comparison_df.empty:
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Chart", "🔥 Heatmap", "📈 Statistical Summary", "📋 Raw Data"])
            
            with tab1:
                # Bar chart
                bar_fig = create_bar_chart(comparison_df, selected_metric)
                if bar_fig:
                    st.plotly_chart(bar_fig, use_container_width=True)
                
            with tab2:
                # Heatmap for all metrics
                heatmap_fig = create_heatmap(filtered_strategies, selected_experiment, available_metrics)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
            
            with tab3:
                # Statistical summary
                create_statistical_summary(comparison_df)
            
            with tab4:
                # Raw data view
                st.markdown('<div class="section-header">📋 Raw Data Comparison</div>', unsafe_allow_html=True)
                
                for strategy in filtered_strategies:
                    with st.expander(f"📊 {strategy} Strategy Data"):
                        data = load_data(strategy, selected_experiment)
                        if data is not None:
                            st.dataframe(data, use_container_width=True)
                            
                            # Quick stats
                            metric_col = metric_mappings[selected_experiment][selected_metric]
                            if metric_col in data.columns:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Mean", f"{data[metric_col].mean():.4f}")
                                with col2:
                                    st.metric("Std Dev", f"{data[metric_col].std():.4f}")
                                with col3:
                                    st.metric("Min", f"{data[metric_col].min():.4f}")
                                with col4:
                                    st.metric("Max", f"{data[metric_col].max():.4f}")
        else:
            st.error("No data available for the selected configuration")
            
    else:
        st.info("👈 Please configure your analysis using the sidebar controls")

else:
    # Model-Approach-Metric Analysis
    if selected_approaches and selected_models and selected_metrics and selected_experiment:
        st.markdown(f"## 🤖 Model-Approach-Metric Analysis: {selected_experiment_label}")
        
        # Create model-approach-metric analysis
        analysis_df = create_model_approach_metric_analysis(
            selected_approaches, selected_experiment, selected_models, selected_metrics
        )
        
        if not analysis_df.empty:
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🔥 Model Heatmap", 
                "🎯 Model-Metric Comparison", 
                "🏆 Approach Rankings",
                "📊 Statistical Deep Dive",
                "🔍 Model Analysis",
                "📋 Raw Analysis Data"
            ])
            
            with tab1:
                # Model Performance Heatmap
                st.markdown('<div class="section-header">🔥 Model Performance Heatmap</div>', unsafe_allow_html=True)
                
                # Option to select metric for heatmap
                heatmap_metric = st.selectbox(
                    "Select metric for heatmap:",
                    options=['Overall'] + selected_metrics,
                    help="Choose specific metric or overall performance"
                )
                
                metric_for_heatmap = None if heatmap_metric == 'Overall' else heatmap_metric
                heatmap_fig = create_model_performance_heatmap(analysis_df, metric_for_heatmap)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
            
            with tab2:
                # Model-Metric Comparison
                st.markdown('<div class="section-header">🎯 Model-Metric Performance Comparison</div>', unsafe_allow_html=True)
                model_metric_fig = create_model_metric_comparison(analysis_df, selected_models)
                if model_metric_fig:
                    st.plotly_chart(model_metric_fig, use_container_width=True)
            
            with tab3:
                # Approach Rankings
                st.markdown('<div class="section-header">🏆 Approach Performance Rankings</div>', unsafe_allow_html=True)
                ranking_result = create_approach_ranking_analysis(analysis_df)
                if ranking_result:
                    ranking_fig, ranking_df = ranking_result
                    st.plotly_chart(ranking_fig, use_container_width=True)
                    
                    st.markdown("### 📋 Detailed Rankings")
                    st.dataframe(ranking_df, use_container_width=True)
            
            with tab4:
                # Statistical Deep Dive
                st.markdown('<div class="section-header">📊 Statistical Deep Dive</div>', unsafe_allow_html=True)
                
                # Best performing combinations
                best_combinations = analysis_df.nlargest(10, 'Mean')[['Model', 'Approach', 'Metric', 'Mean', 'Std']]
                st.markdown("### 🌟 Top 10 Model-Approach-Metric Combinations")
                st.dataframe(best_combinations, use_container_width=True)
                
                # Statistical summary by approach
                st.markdown("### 📈 Performance by Approach")
                approach_stats = analysis_df.groupby('Approach').agg({
                    'Mean': ['mean', 'std', 'min', 'max'],
                    'Std': 'mean'
                }).round(4)
                approach_stats.columns = ['Mean_Avg', 'Mean_Std', 'Min_Performance', 'Max_Performance', 'Avg_Variability']
                st.dataframe(approach_stats, use_container_width=True)
                
                # Model consistency analysis
                st.markdown("### 🎯 Model Consistency Analysis")
                model_consistency = analysis_df.groupby('Model').agg({
                    'Mean': ['mean', 'std'],
                    'Std': 'mean'
                }).round(4)
                model_consistency.columns = ['Avg_Performance', 'Performance_Variability', 'Avg_Std']
                model_consistency['Consistency_Score'] = model_consistency['Avg_Performance'] / (1 + model_consistency['Performance_Variability'])
                model_consistency = model_consistency.sort_values('Consistency_Score', ascending=False)
                st.dataframe(model_consistency, use_container_width=True)
            
            with tab5:
                # Individual Model Analysis
                st.markdown('<div class="section-header">🔍 Individual Model Analysis</div>', unsafe_allow_html=True)
                
                selected_model_analysis = st.selectbox(
                    "Select model for detailed analysis:",
                    options=selected_models
                )
                
                model_data = analysis_df[analysis_df['Model'] == selected_model_analysis]
                
                if not model_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Performance by approach
                        approach_performance = model_data.groupby('Approach')['Mean'].mean().sort_values(ascending=False)
                        
                        fig_approach = px.bar(
                            x=approach_performance.index,
                            y=approach_performance.values,
                            title=f"<b>{selected_model_analysis} - Performance by Approach</b>",
                            labels={'x': 'Approach', 'y': 'Average Performance'},
                            color=approach_performance.values,
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_approach, use_container_width=True)
                    
                    with col2:
                        # Performance by metric
                        metric_performance = model_data.groupby('Metric')['Mean'].mean().sort_values(ascending=False)
                        
                        fig_metric = px.bar(
                            x=metric_performance.index,
                            y=metric_performance.values,
                            title=f"<b>{selected_model_analysis} - Performance by Metric</b>",
                            labels={'x': 'Metric', 'y': 'Average Performance'},
                            color=metric_performance.values,
                            color_continuous_scale='Plasma'
                        )
                        st.plotly_chart(fig_metric, use_container_width=True)
                    
                    # Detailed model stats
                    st.markdown(f"### 📋 {selected_model_analysis} - Detailed Statistics")
                    st.dataframe(model_data, use_container_width=True)
            
            with tab6:
                # Raw Analysis Data
                st.markdown('<div class="section-header">📋 Complete Analysis Dataset</div>', unsafe_allow_html=True)
                st.dataframe(analysis_df, use_container_width=True)
                
                # Summary statistics
                st.markdown("### 📊 Dataset Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Combinations", len(analysis_df))
                with col2:
                    st.metric("Models Analyzed", analysis_df['Model'].nunique())
                with col3:
                    st.metric("Approaches Compared", analysis_df['Approach'].nunique())
                with col4:
                    st.metric("Metrics Evaluated", analysis_df['Metric'].nunique())
        
        else:
            st.error("No data available for the selected model-approach-metric configuration")
    
    else:
        st.info("👈 Please configure your model-approach-metric analysis using the sidebar controls")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Dynamic Benchmark Comparative Analysis Tool</strong></p>
    <p>Comprehensive performance analysis across multiple prompting strategies and experiments</p>
</div>
""", unsafe_allow_html=True)