import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Tuple, Optional

class DataAnalyzer:
    """Dynamic data analyzer that adapts to different dataset structures"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = self._detect_numeric_columns()
        self.categorical_cols = self._detect_categorical_columns()
        self.metric_cols = self._detect_metric_columns()
        self.grouping_cols = self._detect_grouping_columns()
    
    def _detect_numeric_columns(self) -> List[str]:
        """Detect numeric columns including percentages"""
        numeric_cols = []
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                numeric_cols.append(col)
            elif col.strip().endswith(('%', '(%)')) or 'score' in col.lower():
                numeric_cols.append(col)
        return numeric_cols
    
    def _detect_categorical_columns(self) -> List[str]:
        """Detect categorical columns for grouping"""
        categorical_cols = []
        for col in self.df.columns:
            if (self.df[col].dtype == 'object' and 
                self.df[col].nunique() < len(self.df) * 0.5 and
                self.df[col].nunique() > 1):
                categorical_cols.append(col)
        return categorical_cols
    
    def _detect_metric_columns(self) -> List[str]:
        """Detect performance metrics"""
        metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'score', 'error', 'loss', 'auc', 'mae', 'mse']
        metric_cols = []
        
        for col in self.numeric_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in metric_keywords):
                metric_cols.append(col)
            elif col.strip().endswith(('%', '(%)')):
                metric_cols.append(col)
        
        return metric_cols or self.numeric_cols[:5]  # Fallback to first 5 numeric
    
    def _detect_grouping_columns(self) -> List[str]:
        """Detect likely grouping columns like Model, Approach, etc."""
        grouping_keywords = ['model', 'approach', 'method', 'algorithm', 'strategy', 'technique', 'type']
        grouping_cols = []
        
        for col in self.categorical_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in grouping_keywords):
                grouping_cols.append(col)
        
        return grouping_cols or self.categorical_cols[:2]  # Fallback to first 2 categorical

class ModernChartGenerator:
    """Generate modern, interactive charts with creative visualizations"""
    
    def __init__(self, df: pd.DataFrame, analyzer: DataAnalyzer):
        self.df = df
        self.analyzer = analyzer
        self.color_schemes = {
            'modern': px.colors.qualitative.Set3,
            'vibrant': px.colors.qualitative.Vivid,
            'professional': px.colors.qualitative.Safe,
            'gradient': px.colors.sequential.Viridis
        }
    
    def create_radar_comparison(self, metrics: List[str], group_by: str) -> go.Figure:
        """Create radar chart comparing multiple metrics across groups"""
        if not metrics or group_by not in self.df.columns:
            return None
        
        fig = go.Figure()
        
        for group in self.df[group_by].unique():
            group_data = self.df[self.df[group_by] == group]
            values = [group_data[metric].mean() for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=str(group),
                line=dict(width=2),
                fillcolor=f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.3)'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max([self.df[m].max() for m in metrics])])
            ),
            showlegend=True,
            title="Multi-Metric Radar Comparison",
            height=600
        )
        
        return fig
    
    def create_sunburst_hierarchy(self, hierarchy_cols: List[str], metric: str) -> go.Figure:
        """Create sunburst chart showing hierarchical relationships"""
        if len(hierarchy_cols) < 2:
            return None
        
        # Create hierarchical data
        df_agg = self.df.groupby(hierarchy_cols)[metric].mean().reset_index()
        
        fig = go.Figure(go.Sunburst(
            labels=df_agg[hierarchy_cols[0]].tolist() + df_agg[hierarchy_cols[1]].tolist(),
            parents=[""] * len(df_agg[hierarchy_cols[0]].unique()) + df_agg[hierarchy_cols[0]].tolist(),
            values=df_agg[metric].tolist() + df_agg[metric].tolist(),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Value: %{value}<extra></extra>',
            maxdepth=3
        ))
        
        fig.update_layout(
            title=f"Hierarchical {metric} Distribution",
            height=600
        )
        
        return fig
    
    def create_treemap_performance(self, group_cols: List[str], metric: str) -> go.Figure:
        """Create treemap showing performance across different groupings"""
        if not group_cols:
            return None
        
        df_agg = self.df.groupby(group_cols)[metric].agg(['mean', 'count']).reset_index()
        df_agg.columns = group_cols + ['performance', 'sample_count']
        
        fig = go.Figure(go.Treemap(
            labels=df_agg[group_cols[0]].astype(str) + 
                   ((" - " + df_agg[group_cols[1]].astype(str)) if len(group_cols) > 1 else ""),
            values=df_agg['sample_count'],
            parents=[""] * len(df_agg),
            text=df_agg['performance'].round(3),
            texttemplate="<b>%{label}</b><br>Performance: %{text}<br>Samples: %{value}",
            hovertemplate='<b>%{label}</b><br>Performance: %{text}<br>Samples: %{value}<extra></extra>',
            colorscale='RdYlBu',
            colorbar=dict(title="Performance Score")
        ))
        
        fig.update_layout(
            title=f"Performance Treemap by {' & '.join(group_cols)}",
            height=600
        )
        
        return fig
    
    def create_parallel_coordinates(self, metrics: List[str], group_by: str) -> go.Figure:
        """Create parallel coordinates plot for multi-dimensional analysis"""
        if len(metrics) < 2:
            return None
        
        # Normalize metrics to 0-1 scale for better visualization
        df_norm = self.df.copy()
        for metric in metrics:
            df_norm[metric] = (df_norm[metric] - df_norm[metric].min()) / (df_norm[metric].max() - df_norm[metric].min())
        
        # Create dimensions for parallel coordinates
        dimensions = []
        for metric in metrics:
            dimensions.append(dict(
                label=metric,
                values=df_norm[metric]
            ))
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df_norm[group_by].astype('category').cat.codes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=group_by)
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=f"Parallel Coordinates Analysis",
            height=600
        )
        
        return fig
    
    def create_violin_distribution(self, metric: str, group_by: str) -> go.Figure:
        """Create violin plot showing distribution patterns"""
        fig = go.Figure()
        
        for group in self.df[group_by].unique():
            group_data = self.df[self.df[group_by] == group][metric]
            
            fig.add_trace(go.Violin(
                y=group_data,
                name=str(group),
                box_visible=True,
                meanline_visible=True,
                fillcolor=f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.6)',
                line_color='black'
            ))
        
        fig.update_layout(
            title=f"{metric} Distribution by {group_by}",
            yaxis_title=metric,
            xaxis_title=group_by,
            height=500
        )
        
        return fig

def run_python_results_viewer():
    st.markdown("## 🚀 Advanced Python Results Analyzer")
    st.markdown("*Dynamic analysis with modern interactive visualizations*")
    
    base_dir = "data/python_results_data"
    
    # Enhanced sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        view_mode = st.radio("Choose Mode", ["📊 Interactive Analysis", "📂 Dataset Explorer", "🔍 Advanced Metrics"])
        
        color_scheme = st.selectbox("Color Scheme", ["modern", "vibrant", "professional", "gradient"])
        show_stats = st.checkbox("Show Statistics", value=True)
        auto_refresh = st.checkbox("Auto-refresh Charts", value=False)
    
    # Get experiments
    experiments = sorted([
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ])
    
    if not experiments:
        st.error("No experiments found in `python_results_data`.")
        return
    
    # Two-column layout for better UX
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_exp = st.selectbox("🧪 Select Experiment", experiments)
        exp_path = os.path.join(base_dir, selected_exp)
        
        # Get all CSV files
        all_files = []
        for root, dirs, files in os.walk(exp_path):
            for file in files:
                if file.endswith(".csv"):
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    all_files.append(rel_path)
        
        if not all_files:
            st.error("No CSV files found.")
            return
        
        selected_file = st.selectbox("📄 Select Dataset", all_files)
    
    with col2:
        if selected_file:
            file_path = os.path.join(base_dir, selected_file)
            df = pd.read_csv(file_path)
            
            # Initialize analyzer
            analyzer = DataAnalyzer(df)
            chart_gen = ModernChartGenerator(df, analyzer)
            
            # Display dataset info
            st.info(f"**Dataset:** {selected_file} | **Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    if view_mode == "📊 Interactive Analysis":
        # Dynamic control panel
        st.markdown("### 🎯 Interactive Controls")
        
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            primary_metric = st.selectbox("Primary Metric", analyzer.metric_cols)
            
        with control_col2:
            group_by = st.selectbox("Group By", analyzer.grouping_cols)
            
        with control_col3:
            chart_type = st.selectbox("Chart Type", [
                "🎯 Radar Comparison",
                "🌅 Sunburst Hierarchy", 
                "🗺️ Treemap Performance",
                "📊 Parallel Coordinates",
                "🎻 Violin Distribution"
            ])
        
        # Generate selected chart
        st.markdown("### 📈 Interactive Visualization")
        
        if chart_type == "🎯 Radar Comparison":
            selected_metrics = st.multiselect("Select Metrics for Radar", analyzer.metric_cols, default=analyzer.metric_cols[:5])
            if selected_metrics:
                fig = chart_gen.create_radar_comparison(selected_metrics, group_by)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "🌅 Sunburst Hierarchy":
            hierarchy_cols = st.multiselect("Select Hierarchy Columns", analyzer.grouping_cols, default=analyzer.grouping_cols[:2])
            if len(hierarchy_cols) >= 2:
                fig = chart_gen.create_sunburst_hierarchy(hierarchy_cols, primary_metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "🗺️ Treemap Performance":
            group_cols = st.multiselect("Select Grouping Columns", analyzer.grouping_cols, default=analyzer.grouping_cols[:2])
            if group_cols:
                fig = chart_gen.create_treemap_performance(group_cols, primary_metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "📊 Parallel Coordinates":
            parallel_metrics = st.multiselect("Select Metrics", analyzer.metric_cols, default=analyzer.metric_cols[:4])
            if len(parallel_metrics) >= 2:
                fig = chart_gen.create_parallel_coordinates(parallel_metrics, group_by)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "🎻 Violin Distribution":
            fig = chart_gen.create_violin_distribution(primary_metric, group_by)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Interactive model selection
        if group_by in df.columns:
            st.markdown("### 🔧 Model-Specific Analysis")
            unique_groups = df[group_by].unique()
            
            selected_models = st.multiselect(f"Select {group_by}s to Compare", unique_groups, default=unique_groups[:3])
            
            if selected_models:
                filtered_df = df[df[group_by].isin(selected_models)]
                
                # Create comparison chart
                fig = px.bar(
                    filtered_df,
                    x=group_by,
                    y=primary_metric,
                    color=group_by,
                    title=f"{primary_metric} Comparison - Selected Models",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    elif view_mode == "📂 Dataset Explorer":
        st.markdown("### 📊 Dataset Overview")
        
        # Show data preview
        st.dataframe(df.head(), use_container_width=True)
        
        if show_stats:
            st.markdown("### 📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Download functionality
        csv_buffer = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Dataset",
            csv_buffer,
            file_name=f"{selected_exp}_data.csv",
            mime="text/csv"
        )
    
    elif view_mode == "🔍 Advanced Metrics":
        st.markdown("### 🔬 Advanced Metric Analysis")
        
        # Correlation analysis
        if len(analyzer.metric_cols) > 1:
            st.markdown("#### 📊 Metric Correlations")
            corr_matrix = df[analyzer.metric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Metric Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance ranking
        if analyzer.grouping_cols and analyzer.metric_cols:
            st.markdown("#### 🏆 Performance Ranking")
            
            ranking_metric = st.selectbox("Ranking Metric", analyzer.metric_cols)
            ranking_group = st.selectbox("Ranking Group", analyzer.grouping_cols)
            
            ranking_df = df.groupby(ranking_group)[ranking_metric].agg(['mean', 'std', 'count']).reset_index()
            ranking_df = ranking_df.sort_values('mean', ascending=False)
            
            fig = px.bar(
                ranking_df,
                x=ranking_group,
                y='mean',
                error_y='std',
                title=f"Performance Ranking by {ranking_metric}",
                color='mean',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer with dynamic info
    st.markdown("---")
    st.markdown(f"**Detected Structure:** {len(analyzer.metric_cols)} metrics, {len(analyzer.grouping_cols)} grouping columns, {len(analyzer.categorical_cols)} categorical features")
