import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from pathlib import Path

def get_metric_columns():
    """Returns mapping of dataset names to their metric columns"""
    return {
        'NL2PL': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision', 
            'Recall': 'Recall',
            'F1 Score': 'F1',
            'Jaccard Index': 'Jaccard',
            'Levenshtein Distance': 'Levenshtein'
        },
        'NL2FutureLTL': {
            'GT→Pred Accuracy': 'Accuracy_GT_to_Pred (%)',
            'Pred→GT Accuracy': 'Accuracy_Pred_to_GT (%)',
            'Equivalence Accuracy': 'Equivalence_Accuracy (%)',
            'Syntactic Correctness': 'Syntactic_Correctness_Rate (%)',
            'Syntactic Match': 'Syntactic_Match_Rate (%)',
            'Precision': 'Precision (%)',
            'Recall': 'Recall (%)',
            'F1 Score': 'F1 (%)'
        },
        'WFF': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision',
            'Recall': 'Recall',
            'F1 Score': 'F1_Score',
            'True Positives': 'True_Positives',
            'True Negatives': 'True_Negatives',
            'False Positives': 'False_Positives',
            'False Negatives': 'False_Negatives',
        },
        'Trace Generation': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision',
            'Recall': 'Recall',
            'F1 Score': 'F1_Score',
            'Positive Satisfaction Rate': 'Positive_Satisfaction_Rate',
            'Negative Falsification Rate': 'Negative_Falsification_Rate'
        },
        'Trace Characterization': {
            'Accuracy': 'Accuracy',
            'Precision': 'Precision',
            'F1 Score': 'F1'
        },
        'Textbook NL2FutureLTL': {
            'GT→Pred Accuracy': 'Accuracy_GT_to_Pred (%)',
            'Pred→GT Accuracy': 'Accuracy_Pred_to_GT (%)',
            'Equivalence Accuracy': 'Equivalence_Accuracy (%)',
            'Syntactic Correctness': 'Syntactic_Correctness_Rate (%)',
            'Syntactic Match': 'Syntactic_Match_Rate (%)',
            'Precision': 'Precision (%)',
            'Recall': 'Recall (%)',
            'F1 Score': 'F1 (%)'
        },
        'NL2PastLTL': {
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

def get_dataset_file_mapping():
    """Returns mapping of dataset names to their file names"""
    return {
        'NL2FutureLTL': 'comprehensive_table_future_little_tricky.csv',
        'Textbook NL2FutureLTL': 'comprehensive_table_future_textbook.csv',
        'NL2PastLTL': 'comprehensive_table_past_little_tricky.csv',
        'NL2PL': 'nl2pl_aggregated_results.csv',
        'WFF': 'wff_aggregate_metrics.csv',
        'Trace Characterization': 'trace_characterization.csv',
        'Trace Generation': 'trace_generation.csv'
    }

def get_strategy_folder_mapping():
    """Returns mapping of strategy names to their folder names"""
    return {
        'Minimal': 'undefined',
        'Detailed': 'defined',
        'Python': 'python'
    }

def get_available_datasets_for_strategy(strategy):
    """Get available datasets for a specific strategy"""
    folder_mapping = get_strategy_folder_mapping()
    folder = folder_mapping[strategy]
    
    # Special handling for Python strategy (WFF and NL2PL not available)
    if strategy == 'Python':
        excluded_datasets = ['WFF', 'NL2PL']
    else:
        excluded_datasets = []
    
    available_datasets = []
    file_mapping = get_dataset_file_mapping()
    
    # for dataset_name, filename in file_mapping.items():
    #     if dataset_name not in excluded_datasets:
    #         filepath = os.path.join(folder, filename)
    #         if os.path.exists(filepath):
    #             available_datasets.append(dataset_name)
    for dataset_name, filename in file_mapping.items():
        if dataset_name not in excluded_datasets:
            filepath = os.path.join(folder, filename)
            print(f"Checking {filepath}") # <--- NEW!
            if os.path.exists(filepath):
                available_datasets.append(dataset_name)
            else:
                print(f"DATASET MISSING: {filepath}") # <--- NEW!
    return available_datasets

    # return available_datasets

def normalize_model_names(df, strategy):
    """Normalize model names based on strategy"""
    if strategy in ['Minimal', 'Detailed']:
        model_mapping = {
            'claude-sonnet': 'claude-3.5-sonnet',
            'gemini': 'gemini-1.5-flash'
        }
        if 'Model' in df.columns:
            df['Model'] = df['Model'].replace(model_mapping)
    return df

def load_strategy_data(strategy, dataset_name):
    """Load data for a specific strategy and dataset"""
    folder_mapping = get_strategy_folder_mapping()
    file_mapping = get_dataset_file_mapping()
    
    if dataset_name not in file_mapping:
        return None
    
    folder = folder_mapping[strategy]
    filename = file_mapping[dataset_name]
    filepath = os.path.join(folder, filename)
    
    try:
        df = pd.read_csv(filepath)
        df = normalize_model_names(df, strategy)
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading {strategy} data: {str(e)}")
        return None

def create_three_strategy_comparison_chart(data_dict, metric, title):
    """Create a comparison chart for all three strategies"""
    fig = go.Figure()
    
    colors = {
        'Minimal': '#FF6B6B',
        'Detailed': '#4ECDC4', 
        'Python': '#45B7D1'
    }
    
    # Get all unique models across strategies
    all_models = set()
    for strategy, df in data_dict.items():
        if df is not None and 'Model' in df.columns:
            all_models.update(df['Model'].unique())
    
    all_models = sorted(list(all_models))
    
    for strategy, df in data_dict.items():
        if df is not None and metric in df.columns:
            values = []
            error_values = []
            
            for model in all_models:
                model_data = df[df['Model'] == model]
                if not model_data.empty:
                    mean_val = model_data[metric].mean()
                    std_val = model_data[metric].std()
                    values.append(mean_val)
                    error_values.append(std_val if not pd.isna(std_val) else 0)
                else:
                    values.append(None)
                    error_values.append(0)
            
            fig.add_trace(go.Bar(
                name=strategy,
                x=all_models,
                y=values,
                error_y=dict(type='data', array=error_values),
                marker_color=colors.get(strategy, '#95A5A6'),
                text=[f"{v:.2f}" if v is not None else "N/A" for v in values],
                textposition='auto',
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Models",
        yaxis_title=metric,
        barmode='group',
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_strategy_performance_table(data_dict, metrics):
    """Create a performance comparison table"""
    table_data = []
    
    for strategy, df in data_dict.items():
        if df is not None:
            row = {'Strategy': strategy}
            
            for metric in metrics:
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    row[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    row[metric] = "N/A"
            
            table_data.append(row)
    
    return pd.DataFrame(table_data)

def create_model_strategy_heatmap(data_dict, metric, title):
    """Create a heatmap showing model performance across strategies"""
    # Get all unique models
    all_models = set()
    for df in data_dict.values():
        if df is not None and 'Model' in df.columns:
            all_models.update(df['Model'].unique())
    
    all_models = sorted(list(all_models))
    strategies = list(data_dict.keys())
    
    # Create matrix
    matrix = []
    for strategy in strategies:
        row = []
        df = data_dict[strategy]
        for model in all_models:
            if df is not None and metric in df.columns:
                model_data = df[df['Model'] == model]
                if not model_data.empty:
                    row.append(model_data[metric].mean())
                else:
                    row.append(0)
            else:
                row.append(0)
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_models,
        y=strategies,
        colorscale='Viridis',
        text=[[f"{val:.2f}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Models",
        yaxis_title="Strategies",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_metric_comparison_radar(data_dict, metrics, title):
    """Create radar chart comparing strategies across multiple metrics"""
    fig = go.Figure()
    
    colors = {
        'Minimal': '#FF6B6B',
        'Detailed': '#4ECDC4', 
        'Python': '#45B7D1'
    }
    
    for strategy, df in data_dict.items():
        if df is not None:
            values = []
            for metric in metrics:
                if metric in df.columns:
                    values.append(df[metric].mean())
                else:
                    values.append(0)
            
            # Close the radar chart
            values += values[:1]
            metrics_closed = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_closed,
                fill='toself',
                name=strategy,
                line=dict(color=colors.get(strategy, '#95A5A6'))
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([max(values[:-1]) for values in [
                    [df[metric].mean() if metric in df.columns else 0 for metric in metrics]
                    for df in data_dict.values() if df is not None
                ]]) * 1.1]
            )),
        showlegend=True,
        title=title,
        height=500
    )
    
    return fig

def run_benchmark_results():
    """Main function for benchmark results page"""
    st.title("📊 Benchmark Experimental Results")
    st.markdown("### Compare Multiple Prompting Strategies Simultaneously")
    st.markdown("---")
    
    # Three-column layout for strategy selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔴 Minimal Strategy")
        minimal_datasets = get_available_datasets_for_strategy('Minimal')
        minimal_dataset = st.selectbox(
            "Select Dataset (Minimal)",
            minimal_datasets,
            key="minimal_dataset",
            help="Choose dataset for minimal prompting strategy"
        )
    
    with col2:
        st.subheader("🟢 Detailed Strategy")
        detailed_datasets = get_available_datasets_for_strategy('Detailed')
        detailed_dataset = st.selectbox(
            "Select Dataset (Detailed)",
            detailed_datasets,
            key="detailed_dataset",
            help="Choose dataset for detailed prompting strategy"
        )
    
    with col3:
        st.subheader("🔵 Python Strategy")
        python_datasets = get_available_datasets_for_strategy('Python')
        python_dataset = st.selectbox(
            "Select Dataset (Python)",
            python_datasets,
            key="python_dataset",
            help="Choose dataset for python prompting strategy"
        )
    
    # Load data for all three strategies
    data_dict = {}
    selected_datasets = {}
    
    if minimal_dataset:
        minimal_data = load_strategy_data('Minimal', minimal_dataset)
        if minimal_data is not None:
            data_dict['Minimal'] = minimal_data
            selected_datasets['Minimal'] = minimal_dataset
    
    if detailed_dataset:
        detailed_data = load_strategy_data('Detailed', detailed_dataset)
        if detailed_data is not None:
            data_dict['Detailed'] = detailed_data
            selected_datasets['Detailed'] = detailed_dataset
    
    if python_dataset:
        python_data = load_strategy_data('Python', python_dataset)
        if python_data is not None:
            data_dict['Python'] = python_data
            selected_datasets['Python'] = python_dataset
    
    if not data_dict:
        st.error("No data loaded. Please ensure datasets are available and selected.")
        return
    
    # Find common metrics across selected datasets
    metric_columns = get_metric_columns()
    common_metrics = None
    
    for strategy, dataset in selected_datasets.items():
        if dataset in metric_columns:
            dataset_metrics = set(metric_columns[dataset].values())
            if common_metrics is None:
                common_metrics = dataset_metrics
            else:
                common_metrics = common_metrics.intersection(dataset_metrics)
    
    # Filter common metrics to only those that exist in loaded data
    available_metrics = []
    if common_metrics:
        for metric in common_metrics:
            metric_exists = all(
                metric in df.columns for df in data_dict.values() if df is not None
            )
            if metric_exists:
                available_metrics.append(metric)
    
    if not available_metrics:
        st.warning("No common metrics found across selected datasets.")
        return
    
    # Metrics selection
    st.markdown("---")
    st.subheader("📈 Metrics Selection")
    
    selected_metrics = st.multiselect(
        "Select Metrics for Analysis",
        available_metrics,
        default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics,
        help="Choose metrics to compare across strategies"
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric.")
        return
    
    # Display data overview
    st.markdown("---")
    st.subheader("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Strategies", len(data_dict))
    
    with col2:
        total_models = len(set().union(*[
            df['Model'].unique() if 'Model' in df.columns else []
            for df in data_dict.values()
        ]))
        st.metric("Total Unique Models", total_models)
    
    with col3:
        st.metric("Selected Metrics", len(selected_metrics))
    
    with col4:
        if selected_metrics and data_dict:
            avg_performance = np.mean([
                df[selected_metrics[0]].mean() 
                for df in data_dict.values() 
                if selected_metrics[0] in df.columns
            ])
            st.metric(f"Avg {selected_metrics[0]}", f"{avg_performance:.2f}")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategy Comparison", 
        "🎯 Model Performance", 
        "📈 Multi-Metric Analysis",
        "📋 Performance Tables",
        "🔍 Detailed Analysis"
    ])
    
    with tab1:
        st.subheader("Strategy Performance Comparison")
        
        # Individual metric comparisons
        for metric in selected_metrics:
            st.markdown(f"### {metric}")
            
            # Check which strategies have this metric
            strategies_with_metric = {
                strategy: df for strategy, df in data_dict.items()
                if metric in df.columns
            }
            
            if strategies_with_metric:
                fig = create_three_strategy_comparison_chart(
                    strategies_with_metric,
                    metric,
                    f"{metric} Comparison Across Strategies"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show strategy summary
                col1, col2, col3 = st.columns(3)
                for i, (strategy, df) in enumerate(strategies_with_metric.items()):
                    with [col1, col2, col3][i % 3]:
                        mean_val = df[metric].mean()
                        std_val = df[metric].std()
                        st.metric(
                            f"{strategy} {metric}",
                            f"{mean_val:.3f}",
                            f"±{std_val:.3f}"
                        )
            else:
                st.warning(f"No data available for {metric}")
    
    with tab2:
        st.subheader("Model Performance Analysis")
        
        # Model performance heatmap
        for metric in selected_metrics:
            strategies_with_metric = {
                strategy: df for strategy, df in data_dict.items()
                if metric in df.columns
            }
            
            if len(strategies_with_metric) > 1:
                st.markdown(f"### {metric} - Model vs Strategy Heatmap")
                heatmap_fig = create_model_strategy_heatmap(
                    strategies_with_metric,
                    metric,
                    f"{metric} Performance Heatmap"
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Best performing models
        st.markdown("### 🏆 Best Performing Models")
        for metric in selected_metrics:
            best_performers = []
            for strategy, df in data_dict.items():
                if metric in df.columns and 'Model' in df.columns:
                    best_model = df.loc[df[metric].idxmax()]
                    best_performers.append({
                        'Strategy': strategy,
                        'Model': best_model['Model'],
                        'Score': best_model[metric]
                    })
            
            if best_performers:
                best_df = pd.DataFrame(best_performers)
                st.write(f"**{metric} Top Performers:**")
                st.dataframe(best_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Multi-Metric Analysis")
        
        if len(selected_metrics) > 2:
            # Radar chart comparing strategies
            st.markdown("### Strategy Performance Radar")
            radar_fig = create_metric_comparison_radar(
                data_dict,
                selected_metrics,
                "Multi-Metric Strategy Comparison"
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Metric correlations
        st.markdown("### Metric Correlations")
        for strategy, df in data_dict.items():
            if len(selected_metrics) > 1:
                available_metrics_in_df = [m for m in selected_metrics if m in df.columns]
                if len(available_metrics_in_df) > 1:
                    corr_matrix = df[available_metrics_in_df].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.round(2).values,
                        texttemplate="%{text}",
                        textfont={"size": 10}
                    ))
                    
                    fig.update_layout(
                        title=f"{strategy} Strategy - Metric Correlations",
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Performance Summary Tables")
        
        # Overall performance table
        st.markdown("### Overall Strategy Performance")
        performance_table = create_strategy_performance_table(data_dict, selected_metrics)
        st.dataframe(performance_table, use_container_width=True, hide_index=True)
        
        # Detailed model performance
        st.markdown("### Detailed Model Performance")
        for strategy, df in data_dict.items():
            if 'Model' in df.columns:
                st.markdown(f"#### {strategy} Strategy")
                model_summary = df.groupby('Model')[selected_metrics].agg(['mean', 'std']).round(3)
                st.dataframe(model_summary, use_container_width=True)
    
    with tab5:
        st.subheader("Detailed Analysis")
        
        # Strategy insights
        st.markdown("### 🔍 Strategy Insights")
        
        for metric in selected_metrics:
            st.markdown(f"#### {metric} Analysis")
            
            strategy_means = {}
            for strategy, df in data_dict.items():
                if metric in df.columns:
                    strategy_means[strategy] = df[metric].mean()
            
            if strategy_means:
                best_strategy = max(strategy_means, key=strategy_means.get)
                worst_strategy = min(strategy_means, key=strategy_means.get)
                
                st.success(f"🏆 **Best Strategy**: {best_strategy} ({strategy_means[best_strategy]:.3f})")
                st.error(f"📉 **Lowest Strategy**: {worst_strategy} ({strategy_means[worst_strategy]:.3f})")
                
                if len(strategy_means) > 1:
                    performance_gap = strategy_means[best_strategy] - strategy_means[worst_strategy]
                    st.info(f"📊 **Performance Gap**: {performance_gap:.3f} points")
        
        # Raw data display
        if st.checkbox("Show Raw Data"):
            st.markdown("### 📄 Raw Data")
            for strategy, df in data_dict.items():
                st.markdown(f"#### {strategy} Strategy - {selected_datasets.get(strategy, 'Unknown Dataset')}")
                st.dataframe(df, use_container_width=True)

# Integration with existing main function - just add this to your existing main():
# 
# Update your existing main() function's radio button to include "Benchmark Results":
# 
# page = st.sidebar.radio(
#     "Go to",
#     ["Main Dashboard", "Python for Prompt Results", "Benchmark Results"],
#     help="Select the dashboard section"
# )
#
# And add this elif condition:
# elif page == "Benchmark Results":
#     run_benchmark_results()

# Note: Remove the main() function from this file and integrate run_benchmark_results() 
# into your existing streamlit app structure to avoid the set_page_config error.