import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BenchmarkAnalyzer:
    """Comprehensive benchmark analysis system for LLM evaluation"""
    
    def __init__(self):
        self.base_dir = "compare_data"
        self.task_metrics = self._get_task_metrics()
        self.model_mappings = self._get_model_mappings()
        self.color_palette = px.colors.qualitative.Vivid
        
    def _get_task_metrics(self) -> Dict[str, Dict[str, str]]:
        """Define metrics for each task"""
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
    
    def _get_model_mappings(self) -> Dict[str, Dict[str, str]]:
        """Define model name mappings for different prompting strategies"""
        return {
            'minimal': {
                'claude-sonnet': 'claude-3.5-sonnet',
                'gemini': 'gemini-1.5-flash'
            },
            'detailed': {
                'claude-sonnet': 'claude-3.5-sonnet',
                'gemini': 'gemini-1.5-flash'
            },
            'python': {}  # Python uses actual model names
        }
    
    def get_file_mapping(self) -> Dict[str, str]:
        """Map task names to file names"""
        return {
            'NL2FutureLTL': 'comprehensive_table_future_little_tricky.csv',
            'Textbook NL2FutureLTL': 'comprehensive_table_future_textbook.csv',
            'NL2PastLTL': 'comprehensive_table_past_little_tricky.csv',
            'NL2PL': 'nl2pl_aggregated_results.csv',
            'WFF': 'wff_aggregate_metrics.csv',
            'Trace Characterization': 'trace_characterization.csv',
            'Trace Generation': 'trace_generation.csv'
        }
    
    def get_available_tasks(self, strategy: str) -> List[str]:
        """Get available tasks for a given strategy"""
        strategy_dir = os.path.join(self.base_dir, strategy)
        if not os.path.exists(strategy_dir):
            return []
        
        file_mapping = self.get_file_mapping()
        available_tasks = []
        
        for task, filename in file_mapping.items():
            if strategy == 'python' and task in ['WFF', 'NL2PL']:
                continue  # These tasks weren't conducted in python
            if os.path.exists(os.path.join(strategy_dir, filename)):
                available_tasks.append(task)
        
        return available_tasks
    
    def load_data(self, strategy: str, task: str) -> Optional[pd.DataFrame]:
        """Load data for a specific strategy and task"""
        file_mapping = self.get_file_mapping()
        
        if task not in file_mapping:
            return None
        
        file_path = os.path.join(self.base_dir, strategy, file_mapping[task])
        
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            # Apply model name mappings
            if 'Model' in df.columns and strategy in self.model_mappings:
                mapping = self.model_mappings[strategy]
                df['Model'] = df['Model'].replace(mapping)
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def calculate_statistical_summary(self, data_dict: Dict[str, pd.DataFrame], metric: str) -> Dict:
        """Calculate comprehensive statistical summary across strategies"""
        summary = {}
        
        # Extract metric values for each strategy
        strategy_data = {}
        for strategy, df in data_dict.items():
            if df is not None and metric in df.columns:
                strategy_data[strategy] = df[metric].dropna()
        
        if len(strategy_data) < 2:
            return summary
        
        # Calculate basic statistics
        for strategy, values in strategy_data.items():
            summary[f"{strategy}_mean"] = values.mean()
            summary[f"{strategy}_std"] = values.std()
            summary[f"{strategy}_median"] = values.median()
            summary[f"{strategy}_count"] = len(values)
        
        # Calculate improvements
        strategies = list(strategy_data.keys())
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                strategy1, strategy2 = strategies[i], strategies[j]
                
                mean1 = strategy_data[strategy1].mean()
                mean2 = strategy_data[strategy2].mean()
                
                improvement = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0
                summary[f"{strategy2}_vs_{strategy1}_improvement"] = improvement
                
                # Statistical significance test
                try:
                    stat, p_value = stats.ttest_ind(strategy_data[strategy1], strategy_data[strategy2])
                    summary[f"{strategy2}_vs_{strategy1}_p_value"] = p_value
                    summary[f"{strategy2}_vs_{strategy1}_significant"] = p_value < 0.05
                except:
                    summary[f"{strategy2}_vs_{strategy1}_p_value"] = None
                    summary[f"{strategy2}_vs_{strategy1}_significant"] = False
        
        return summary
    
    def create_comparison_bar_chart(self, data_dict: Dict[str, pd.DataFrame], metric: str, task: str) -> go.Figure:
        """Create publication-worthy comparison bar chart"""
        fig = make_subplots(
            rows=1, cols=len(data_dict),
            subplot_titles=list(data_dict.keys()),
            shared_yaxes=True
        )
        
        colors = px.colors.qualitative.Set3
        
        for idx, (strategy, df) in enumerate(data_dict.items()):
            if df is not None and metric in df.columns:
                # Group by model if available
                if 'Model' in df.columns:
                    grouped = df.groupby('Model')[metric].agg(['mean', 'std']).reset_index()
                    
                    fig.add_trace(
                        go.Bar(
                            x=grouped['Model'],
                            y=grouped['mean'],
                            error_y=dict(type='data', array=grouped['std']),
                            name=strategy,
                            marker_color=colors[idx % len(colors)],
                            showlegend=(idx == 0)
                        ),
                        row=1, col=idx + 1
                    )
                else:
                    # If no model column, show overall distribution
                    fig.add_trace(
                        go.Bar(
                            x=[strategy],
                            y=[df[metric].mean()],
                            error_y=dict(type='data', array=[df[metric].std()]),
                            name=strategy,
                            marker_color=colors[idx % len(colors)],
                            showlegend=(idx == 0)
                        ),
                        row=1, col=idx + 1
                    )
        
        fig.update_layout(
            title=f"{task} - {metric} Comparison Across Strategies",
            height=500,
            showlegend=False
        )
        
        return fig
    
    # def create_radar_chart(self, data_dict: Dict[str, pd.DataFrame], metrics: List[str], task: str) -> go.Figure:
    #     """Create radar chart for multi-metric comparison"""
    #     fig = go.Figure()
        
    #     colors = px.colors.qualitative.Set3
        
    #     for idx, (strategy, df) in enumerate(data_dict.items()):
    #         if df is not None:
    #             values = []
    #             for metric in metrics:
    #                 if metric in df.columns:
    #                     values.append(df[metric].mean())
    #                 else:
    #                     values.append(0)
                
    #             fig.add_trace(go.Scatterpolar(
    #                 r=values,
    #                 theta=metrics,
    #                 fill='toself',
    #                 name=strategy.title(),
    #                 line=dict(color=colors[idx % len(colors)], width=2),
    #                 fillcolor=f'rgba({int(colors[idx % len(colors)][4:6], 16)}, {int(colors[idx % len(colors)][6:8], 16)}, {int(colors[idx % len(colors)][8:10], 16)}, 0.3)'
    #             ))
        
    #     fig.update_layout(
    #         polar=dict(
    #             radialaxis=dict(
    #                 visible=True,
    #                 range=[0, max([
    #                     max([df[m].mean() for m in metrics if m in df.columns]) 
    #                     for df in data_dict.values() if df is not None
    #                 ]) * 1.1]
    #             )
    #         ),
    #         showlegend=True,
    #         title=f"{task} - Multi-Metric Radar Comparison",
    #         height=600
    #     )
        
    #     return fig
    
    def create_performance_heatmap(self, data_dict: Dict[str, pd.DataFrame], metrics: List[str], task: str) -> go.Figure:
        """Create heatmap showing performance across strategies and metrics"""
        # Prepare data for heatmap
        heatmap_data = []
        strategies = []
        
        for strategy, df in data_dict.items():
            if df is not None:
                strategies.append(strategy.title())
                row_data = []
                for metric in metrics:
                    if metric in df.columns:
                        row_data.append(df[metric].mean())
                    else:
                        row_data.append(0)
                heatmap_data.append(row_data)
        
        if not heatmap_data:
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metrics,
            y=strategies,
            colorscale='Viridis',
            text=np.round(heatmap_data, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"{task} - Performance Heatmap",
            height=400,
            xaxis_title="Metrics",
            yaxis_title="Strategies"
        )
        
        return fig
    
    def create_model_comparison_line(self, data_dict: Dict[str, pd.DataFrame], metric: str, task: str) -> go.Figure:
        """Create line chart comparing models across strategies"""
        fig = go.Figure()
        
        # Get all unique models across strategies
        all_models = set()
        for df in data_dict.values():
            if df is not None and 'Model' in df.columns:
                all_models.update(df['Model'].unique())
        
        colors = px.colors.qualitative.Set3
        
        for idx, model in enumerate(sorted(all_models)):
            x_vals = []
            y_vals = []
            
            for strategy, df in data_dict.items():
                if df is not None and 'Model' in df.columns and metric in df.columns:
                    model_data = df[df['Model'] == model]
                    if not model_data.empty:
                        x_vals.append(strategy.title())
                        y_vals.append(model_data[metric].mean())
            
            if x_vals:
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=model,
                    line=dict(color=colors[idx % len(colors)], width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=f"{task} - {metric} Across Strategies by Model",
            xaxis_title="Strategy",
            yaxis_title=metric,
            height=500,
            hovermode='x unified'
        )
        
        return fig

def run_benchmark_analysis():
    """Main benchmark analysis dashboard"""
    st.markdown("# 🎯 Benchmark Analysis Dashboard")
    st.markdown("*Comprehensive analysis of LLM performance across different prompting strategies*")
    
    analyzer = BenchmarkAnalyzer()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Analysis Controls")
        
        # Strategy selection
        strategies = ['Minimal', 'Detailed', 'Python']
        selected_strategies = st.multiselect(
            "Select Prompting Strategies",
            strategies,
            default=['Minimal', 'Detailed']
        )
        
        # Task selection
        if selected_strategies:
            # Get tasks available for all selected strategies
            available_tasks = set(analyzer.get_available_tasks(selected_strategies[0]))
            for strategy in selected_strategies[1:]:
                available_tasks &= set(analyzer.get_available_tasks(strategy))
            
            available_tasks = sorted(list(available_tasks))
            
            if available_tasks:
                selected_task = st.selectbox("Select Task", available_tasks)
                
                # Load data for selected strategies and task
                data_dict = {}
                for strategy in selected_strategies:
                    data_dict[strategy] = analyzer.load_data(strategy, selected_task)
                
                # Get available metrics for the selected task
                if selected_task in analyzer.task_metrics:
                    available_metrics = list(analyzer.task_metrics[selected_task].values())
                    
                    # Filter metrics that exist in the data
                    valid_metrics = []
                    for strategy_df in data_dict.values():
                        if strategy_df is not None:
                            valid_metrics.extend([col for col in strategy_df.columns if col in available_metrics])
                    
                    valid_metrics = sorted(list(set(valid_metrics)))
                    
                    if valid_metrics:
                        selected_metric = st.selectbox("Select Primary Metric", valid_metrics)
                        
                        # Chart type selection
                        chart_types = [
                            "📊 Comparison Bar Chart",
                            # "🎯 Multi-Metric Radar",
                            "🔥 Performance Heatmap",
                            "📈 Model Comparison Line",
                            "📊 Statistical Summary"
                        ]
                        
                        selected_charts = st.multiselect(
                            "Select Visualizations",
                            chart_types,
                            default=["📊 Comparison Bar Chart", "📊 Statistical Summary"]
                        )
    
    # Main content area
    if 'selected_strategies' in locals() and selected_strategies and 'selected_task' in locals():
        # Display task information
        st.markdown(f"## 📋 Task: {selected_task}")
        
        # Show data overview
        with st.expander("📊 Data Overview"):
            for strategy, df in data_dict.items():
                if df is not None:
                    st.markdown(f"### {strategy.title()} Strategy")
                    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                    st.dataframe(df.head(3))
        
        # Generate visualizations
        if 'selected_charts' in locals():
            for chart_type in selected_charts:
                if chart_type == "📊 Comparison Bar Chart":
                    st.markdown("### 📊 Strategy Comparison")
                    fig = analyzer.create_comparison_bar_chart(data_dict, selected_metric, selected_task)
                    st.plotly_chart(fig, use_container_width=True)
                
                # elif chart_type == "🎯 Multi-Metric Radar":
                #     st.markdown("### 🎯 Multi-Metric Radar Analysis")
                #     # Select multiple metrics for radar chart
                #     radar_metrics = st.multiselect(
                #         "Select Metrics for Radar Chart",
                #         valid_metrics,
                #         default=valid_metrics[:5]
                #     )
                    
                #     if radar_metrics:
                #         fig = analyzer.create_radar_chart(data_dict, radar_metrics, selected_task)
                #         st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "🔥 Performance Heatmap":
                    st.markdown("### 🔥 Performance Heatmap")
                    fig = analyzer.create_performance_heatmap(data_dict, valid_metrics, selected_task)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "📈 Model Comparison Line":
                    st.markdown("### 📈 Model Performance Trends")
                    fig = analyzer.create_model_comparison_line(data_dict, selected_metric, selected_task)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "📊 Statistical Summary":
                    st.markdown("### 📊 Statistical Analysis & Insights")
                    
                    # Calculate statistical summary
                    stats_summary = analyzer.calculate_statistical_summary(data_dict, selected_metric)
                    
                    if stats_summary:
                        # Create columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Performance Summary")
                            for strategy in selected_strategies:
                                if f"{strategy}_mean" in stats_summary:
                                    mean_val = stats_summary[f"{strategy}_mean"]
                                    std_val = stats_summary[f"{strategy}_std"]
                                    count_val = stats_summary[f"{strategy}_count"]
                                    
                                    st.metric(
                                        f"{strategy.title()} Strategy",
                                        f"{mean_val:.3f} ± {std_val:.3f}",
                                        f"n={count_val}"
                                    )
                        
                        with col2:
                            st.markdown("#### 🔍 Strategy Comparisons")
                            for key, value in stats_summary.items():
                                if "_vs_" in key and "_improvement" in key:
                                    strategies_compared = key.replace("_improvement", "").split("_vs_")
                                    improvement = value
                                    
                                    # Get significance
                                    sig_key = key.replace("_improvement", "_significant")
                                    is_significant = stats_summary.get(sig_key, False)
                                    
                                    color = "green" if improvement > 0 else "red"
                                    significance = "✓" if is_significant else "✗"
                                    
                                    st.markdown(
                                        f"**{strategies_compared[0].title()} → {strategies_compared[1].title()}:** "
                                        f"<span style='color:{color}'>{improvement:+.1f}%</span> "
                                        f"(Significant: {significance})",
                                        unsafe_allow_html=True
                                    )
                        
                        # Key insights
                        st.markdown("#### 💡 Key Insights")
                        insights = []
                        
                        # Find best performing strategy
                        best_strategy = max(selected_strategies, 
                                          key=lambda s: stats_summary.get(f"{s}_mean", 0))
                        insights.append(f"🏆 **Best performing strategy:** {best_strategy.title()}")
                        
                        # Find largest improvement
                        improvements = {k: v for k, v in stats_summary.items() if "_improvement" in k}
                        if improvements:
                            best_improvement = max(improvements.items(), key=lambda x: x[1])
                            strategies_compared = best_improvement[0].replace("_improvement", "").split("_vs_")
                            insights.append(
                                f"📈 **Largest improvement:** {strategies_compared[0].title()} → "
                                f"{strategies_compared[1].title()} (+{best_improvement[1]:.1f}%)"
                            )
                        
                        # Statistical significance summary
                        significant_comparisons = [k for k, v in stats_summary.items() 
                                                 if "_significant" in k and v]
                        insights.append(f"📊 **Statistically significant comparisons:** {len(significant_comparisons)}")
                        
                        for insight in insights:
                            st.markdown(insight)
    
    else:
        st.info("👈 Please select prompting strategies and a task from the sidebar to begin analysis.")
