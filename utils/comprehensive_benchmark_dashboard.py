import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BenchmarkDataManager:
    """Manages benchmark data across different prompting strategies and tasks"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.strategy_mapping = {
            "MINIMAL": "undefined",
            "DETAILED": "defined", 
            "PYTHON": "python"
        }
        
        self.task_file_mapping = {
            'NL2FutureLTL': 'comprehensive_table_future_little_tricky.csv',
            'Textbook NL2FutureLTL': 'comprehensive_table_future_textbook.csv',
            'NL2PastLTL': 'comprehensive_table_past_little_tricky.csv',
            'NL2PL': 'nl2pl_aggregated_results.csv',
            'WFF': 'wff_aggregate_metrics.csv',
            'Trace Characterization': 'trace_characterization.csv',
            'Trace Generation': 'trace_generation.csv'
        }
        
        self.metric_mappings = {
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
        
        self.model_name_mapping = {
            'claude-sonnet': 'claude-3.5-sonnet',
            'gemini': 'gemini-1.5-flash'
        }
        
        # Tasks not available in Python strategy
        self.python_excluded_tasks = ['WFF', 'NL2PL']
    
    def load_data(self, strategy: str, task: str) -> Optional[pd.DataFrame]:
        """Load data for specific strategy and task"""
        if strategy == "PYTHON" and task in self.python_excluded_tasks:
            return None
        
        folder = self.strategy_mapping[strategy]
        filename = self.task_file_mapping[task]
        filepath = os.path.join(self.base_dir, folder, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Normalize model names for non-Python strategies
            if strategy in ["MINIMAL", "DETAILED"] and 'Model' in df.columns:
                df['Model'] = df['Model'].map(self.model_name_mapping).fillna(df['Model'])
            
            return df
        except Exception as e:
            st.error(f"Error loading {filepath}: {str(e)}")
            return None
    
    def get_available_tasks(self, strategy: str) -> List[str]:
        """Get available tasks for a strategy"""
        if strategy == "PYTHON":
            return [task for task in self.task_file_mapping.keys() 
                   if task not in self.python_excluded_tasks]
        return list(self.task_file_mapping.keys())
    
    def get_metrics(self, task: str) -> Dict[str, str]:
        """Get available metrics for a task"""
        return self.metric_mappings.get(task, {})

class PublicationChartGenerator:
    """Generate publication-worthy charts with advanced analytics"""
    
    def __init__(self):
        self.color_palettes = {
            'models': px.colors.qualitative.Set1,
            'strategies': ['#FF6B6B', '#4ECDC4', '#45B7D1'],
            'metrics': px.colors.qualitative.Set3,
            'sequential': px.colors.sequential.Viridis
        }
        
        self.chart_style = {
            'font_family': 'Arial, sans-serif',
            'font_size': 12,
            'title_font_size': 16,
            'axis_font_size': 11,
            'legend_font_size': 10
        }
    
    def create_grouped_bar_chart(self, df: pd.DataFrame, metric: str, 
                                group_by: str = 'Model', title: str = None) -> go.Figure:
        """Create publication-quality grouped bar chart"""
        fig = px.bar(
            df, 
            x=group_by, 
            y=metric,
            color=group_by,
            title=title or f"{metric} by {group_by}",
            color_discrete_sequence=self.color_palettes['models']
        )
        
        # Add value labels on bars
        for trace in fig.data:
            trace.texttemplate = '%{y:.2f}'
            trace.textposition = 'outside'
        
        fig.update_layout(
            font_family=self.chart_style['font_family'],
            font_size=self.chart_style['font_size'],
            title_font_size=self.chart_style['title_font_size'],
            showlegend=True,
            height=500,
            xaxis_title=group_by,
            yaxis_title=metric,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_strategy_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                                 metric: str, task: str) -> go.Figure:
        """Create strategy comparison chart"""
        fig = go.Figure()
        
        strategies = list(data_dict.keys())
        x_pos = np.arange(len(strategies))
        
        # Get unique models across all strategies
        all_models = set()
        for df in data_dict.values():
            if 'Model' in df.columns:
                all_models.update(df['Model'].unique())
        
        all_models = sorted(list(all_models))
        
        # Create grouped bars
        bar_width = 0.8 / len(all_models)
        
        for i, model in enumerate(all_models):
            values = []
            for strategy in strategies:
                df = data_dict[strategy]
                if df is not None and 'Model' in df.columns:
                    model_data = df[df['Model'] == model]
                    if not model_data.empty and metric in model_data.columns:
                        values.append(model_data[metric].mean())
                    else:
                        values.append(0)
                else:
                    values.append(0)
            
            fig.add_trace(go.Bar(
                name=model,
                x=[pos + i * bar_width for pos in x_pos],
                y=values,
                width=bar_width,
                text=[f'{v:.2f}' for v in values],
                textposition='outside',
                marker_color=self.color_palettes['models'][i % len(self.color_palettes['models'])]
            ))
        
        fig.update_layout(
            title=f'{task}: {metric} Comparison Across Strategies',
            xaxis_title='Prompting Strategy',
            yaxis_title=metric,
            xaxis=dict(
                tickmode='array',
                tickvals=x_pos,
                ticktext=strategies
            ),
            barmode='group',
            height=500,
            font_family=self.chart_style['font_family'],
            font_size=self.chart_style['font_size'],
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_heatmap(self, df: pd.DataFrame, metrics: List[str]) -> go.Figure:
        """Create correlation heatmap"""
        if len(metrics) < 2:
            return None
        
        # Filter for available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        if len(available_metrics) < 2:
            return None
        
        corr_matrix = df[available_metrics].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Metric Correlation Matrix',
            height=500,
            font_family=self.chart_style['font_family'],
            font_size=self.chart_style['font_size']
        )
        
        return fig
    
    def create_radar_chart(self, df: pd.DataFrame, metrics: List[str], 
                          group_by: str = 'Model') -> go.Figure:
        """Create radar chart for multi-metric comparison"""
        if len(metrics) < 3:
            return None
        
        # Filter for available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        if len(available_metrics) < 3:
            return None
        
        fig = go.Figure()
        
        for group in df[group_by].unique():
            group_data = df[df[group_by] == group]
            values = [group_data[metric].mean() for metric in available_metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics,
                fill='toself',
                name=str(group),
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([df[m].max() for m in available_metrics if m in df.columns])]
                )
            ),
            showlegend=True,
            title=f"Multi-Metric Radar Comparison by {group_by}",
            height=600,
            font_family=self.chart_style['font_family']
        )
        
        return fig
    
    def create_performance_matrix(self, data_dict: Dict[str, pd.DataFrame], 
                                metrics: List[str], task: str) -> go.Figure:
        """Create performance matrix across strategies and models"""
        # Prepare data for matrix
        matrix_data = []
        strategies = list(data_dict.keys())
        
        # Get all unique models
        all_models = set()
        for df in data_dict.values():
            if df is not None and 'Model' in df.columns:
                all_models.update(df['Model'].unique())
        
        all_models = sorted(list(all_models))
        
        # Create matrix for first metric
        if not metrics:
            return None
        
        primary_metric = metrics[0]
        
        matrix = []
        for model in all_models:
            row = []
            for strategy in strategies:
                df = data_dict[strategy]
                if df is not None and 'Model' in df.columns and primary_metric in df.columns:
                    model_data = df[df['Model'] == model]
                    if not model_data.empty:
                        row.append(model_data[primary_metric].mean())
                    else:
                        row.append(0)
                else:
                    row.append(0)
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=strategies,
            y=all_models,
            colorscale='Viridis',
            text=[[f'{val:.2f}' for val in row] for row in matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title=primary_metric)
        ))
        
        fig.update_layout(
            title=f'{task}: {primary_metric} Performance Matrix',
            xaxis_title='Prompting Strategy',
            yaxis_title='Model',
            height=400,
            font_family=self.chart_style['font_family']
        )
        
        return fig

def run_benchmark_dashboard():
    """Main benchmark dashboard function"""
    st.markdown("# 🏆 Benchmark Analysis Dashboard")
    st.markdown("*Comprehensive analysis of prompting strategies across multiple tasks*")
    
    # Initialize data manager and chart generator
    data_manager = BenchmarkDataManager()
    chart_gen = PublicationChartGenerator()
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        # Strategy selection
        strategy = st.selectbox(
            "🎯 Prompting Strategy",
            ["MINIMAL", "DETAILED", "PYTHON"],
            help="Select the prompting strategy to analyze"
        )
        
        # Task selection based on strategy
        available_tasks = data_manager.get_available_tasks(strategy)
        task = st.selectbox(
            "📋 Benchmark Task",
            available_tasks,
            help="Select the benchmark task to analyze"
        )
        
        # Load data
        df = data_manager.load_data(strategy, task)
        
        if df is not None:
            # Metric selection
            available_metrics = data_manager.get_metrics(task)
            if available_metrics:
                metric_options = list(available_metrics.keys())
                selected_metric_name = st.selectbox(
                    "📈 Primary Metric",
                    metric_options,
                    help="Select the primary metric for analysis"
                )
                selected_metric = available_metrics[selected_metric_name]
            else:
                st.warning("No metrics defined for this task")
                return
            
            # Chart type selection
            chart_type = st.selectbox(
                "📊 Chart Type",
                ["🎯 Multi-Strategy Comparison", 
                 "🔥 Performance Heatmap", "🎭 Radar Analysis", "📈 Correlation Matrix"],
                help="Select the type of visualization"
            )
            
            # Additional options
            st.subheader("🎨 Display Options")
            show_stats = st.checkbox("Show Statistics", value=True)
            show_data_preview = st.checkbox("Show Data Preview", value=False)
        else:
            st.error(f"No data available for {strategy} strategy with {task} task")
            return
    
    # Main content area
    if df is not None:
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", f"{df.shape[0]} rows")
        with col2:
            st.metric("Features", f"{df.shape[1]} columns")
        with col3:
            if 'Model' in df.columns:
                st.metric("Models", f"{df['Model'].nunique()} unique")
        
        # Chart generation
        st.markdown("## 📈 Visualization")
        
        if chart_type == "📊 Grouped Bar Chart":
            if selected_metric in df.columns:
                fig = chart_gen.create_grouped_bar_chart(
                    df, selected_metric, 'Model', 
                    f"{task}: {selected_metric_name} by Model ({strategy} Strategy)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Metric '{selected_metric}' not found in dataset")
        
        elif chart_type == "🎯 Multi-Strategy Comparison":
            # Load data for all strategies
            strategy_data = {}
            for strat in ["MINIMAL", "DETAILED", "PYTHON"]:
                if task in data_manager.get_available_tasks(strat):
                    strategy_data[strat] = data_manager.load_data(strat, task)
            
            if len(strategy_data) > 1:
                fig = chart_gen.create_strategy_comparison(
                    strategy_data, selected_metric, task
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough strategies available for comparison")
        
        elif chart_type == "🔥 Performance Heatmap":
            # Load data for all strategies
            strategy_data = {}
            for strat in ["MINIMAL", "DETAILED", "PYTHON"]:
                if task in data_manager.get_available_tasks(strat):
                    strategy_data[strat] = data_manager.load_data(strat, task)
            
            if len(strategy_data) > 1:
                fig = chart_gen.create_performance_matrix(
                    strategy_data, [selected_metric], task
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough strategies available for heatmap")
        
        elif chart_type == "🎭 Radar Analysis":
            available_metrics = data_manager.get_metrics(task)
            metric_columns = [available_metrics[k] for k in available_metrics.keys() 
                            if available_metrics[k] in df.columns]
            
            if len(metric_columns) >= 3:
                fig = chart_gen.create_radar_chart(df, metric_columns, 'Model')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 3 metrics for radar chart")
        
        elif chart_type == "📈 Correlation Matrix":
            available_metrics = data_manager.get_metrics(task)
            metric_columns = [available_metrics[k] for k in available_metrics.keys() 
                            if available_metrics[k] in df.columns]
            
            if len(metric_columns) >= 2:
                fig = chart_gen.create_heatmap(df, metric_columns)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 metrics for correlation matrix")
        
        # Statistics section
        if show_stats:
            st.markdown("## 📊 Statistical Summary")
            
            # Performance summary
            if 'Model' in df.columns and selected_metric in df.columns:
                summary_df = df.groupby('Model')[selected_metric].agg(['mean', 'std', 'count']).round(3)
                summary_df.columns = ['Mean', 'Std Dev', 'Count']
                st.dataframe(summary_df, use_container_width=True)
            
            # Overall statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("### Overall Statistics")
                st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
        
        # Data preview
        if show_data_preview:
            st.markdown("## 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download option
            csv_buffer = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Dataset",
                csv_buffer,
                file_name=f"{task}_{strategy}_data.csv",
                mime="text/csv"
            )

