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

class DatasetComparator:
    """Compare two datasets with similar column structures"""
    
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
        self.df1 = df1
        self.df2 = df2
        self.name1 = name1
        self.name2 = name2
        self.common_columns = self._find_common_columns()
        self.analyzer1 = DataAnalyzer(df1)
        self.analyzer2 = DataAnalyzer(df2)
    
    def _find_common_columns(self) -> Dict[str, List[str]]:
        """Find common columns between datasets"""
        common_cols = set(self.df1.columns) & set(self.df2.columns)
        
        common_numeric = [col for col in common_cols if 
                         self.df1[col].dtype in ['float64', 'int64', 'float32', 'int32'] and
                         self.df2[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        common_categorical = [col for col in common_cols if 
                             self.df1[col].dtype == 'object' and 
                             self.df2[col].dtype == 'object']
        
        return {
            'numeric': common_numeric,
            'categorical': common_categorical,
            'all': list(common_cols)
        }
    
    def create_comparison_bar_chart(self, metric: str, group_by: str) -> go.Figure:
        """Create side-by-side comparison bar chart"""
        if metric not in self.common_columns['numeric'] or group_by not in self.common_columns['categorical']:
            return None
        
        # Aggregate data for both datasets
        agg1 = self.df1.groupby(group_by)[metric].agg(['mean', 'std', 'count']).reset_index()
        agg2 = self.df2.groupby(group_by)[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": False}]]
        )
        
        # Add bars for dataset 1
        fig.add_trace(
            go.Bar(
                name=self.name1,
                x=agg1[group_by],
                y=agg1['mean'],
                error_y=dict(type='data', array=agg1['std']),
                marker_color='rgba(55, 128, 191, 0.8)',
                text=agg1['mean'].round(3),
                textposition='outside',
                hovertemplate=f'<b>{self.name1}</b><br>%{{x}}: %{{y:.3f}}<br>Std: %{{error_y.array:.3f}}<br>Count: %{{customdata}}<extra></extra>',
                customdata=agg1['count']
            )
        )
        
        # Add bars for dataset 2
        fig.add_trace(
            go.Bar(
                name=self.name2,
                x=agg2[group_by],
                y=agg2['mean'],
                error_y=dict(type='data', array=agg2['std']),
                marker_color='rgba(255, 127, 14, 0.8)',
                text=agg2['mean'].round(3),
                textposition='outside',
                hovertemplate=f'<b>{self.name2}</b><br>%{{x}}: %{{y:.3f}}<br>Std: %{{error_y.array:.3f}}<br>Count: %{{customdata}}<extra></extra>',
                customdata=agg2['count']
            )
        )
        
        fig.update_layout(
            title=f"{metric} Comparison: {self.name1} vs {self.name2}",
            xaxis_title=group_by,
            yaxis_title=metric,
            barmode='group',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_delta_chart(self, metric: str, group_by: str) -> go.Figure:
        """Create chart showing performance differences between datasets"""
        if metric not in self.common_columns['numeric'] or group_by not in self.common_columns['categorical']:
            return None
        
        agg1 = self.df1.groupby(group_by)[metric].mean().reset_index()
        agg2 = self.df2.groupby(group_by)[metric].mean().reset_index()
        
        # Merge and calculate delta
        merged = pd.merge(agg1, agg2, on=group_by, suffixes=('_1', '_2'))
        merged['delta'] = merged[f'{metric}_2'] - merged[f'{metric}_1']
        merged['delta_pct'] = (merged['delta'] / merged[f'{metric}_1']) * 100
        
        # Create waterfall-style chart
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in merged['delta']]
        
        fig.add_trace(
            go.Bar(
                x=merged[group_by],
                y=merged['delta'],
                marker_color=colors,
                text=merged['delta'].round(3),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Δ: %{y:.3f}<br>%Change: %{customdata:.1f}%<extra></extra>',
                customdata=merged['delta_pct']
            )
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f"{metric} Performance Delta: {self.name2} - {self.name1}",
            xaxis_title=group_by,
            yaxis_title=f"Δ {metric}",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_correlation_comparison(self) -> go.Figure:
        """Compare correlation patterns between datasets"""
        if len(self.common_columns['numeric']) < 2:
            return None
        
        corr1 = self.df1[self.common_columns['numeric']].corr()
        corr2 = self.df2[self.common_columns['numeric']].corr()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"{self.name1} Correlations", f"{self.name2} Correlations"],
            horizontal_spacing=0.1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=corr1.values,
                x=corr1.columns,
                y=corr1.columns,
                colorscale='RdBu',
                zmid=0,
                showscale=False,
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=corr2.values,
                x=corr2.columns,
                y=corr2.columns,
                colorscale='RdBu',
                zmid=0,
                showscale=True,
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Correlation Pattern Comparison",
            height=500
        )
        
        return fig
    
    def create_distribution_comparison(self, metric: str, group_by: str) -> go.Figure:
        """Create overlapping distribution comparison"""
        if metric not in self.common_columns['numeric'] or group_by not in self.common_columns['categorical']:
            return None
        
        fig = go.Figure()
        
        # Get unique groups from both datasets
        groups1 = set(self.df1[group_by].unique())
        groups2 = set(self.df2[group_by].unique())
        common_groups = groups1 & groups2
        
        for group in common_groups:
            data1 = self.df1[self.df1[group_by] == group][metric]
            data2 = self.df2[self.df2[group_by] == group][metric]
            
            fig.add_trace(go.Histogram(
                x=data1,
                name=f"{group} - {self.name1}",
                opacity=0.7,
                nbinsx=20,
                histnorm='probability density'
            ))
            
            fig.add_trace(go.Histogram(
                x=data2,
                name=f"{group} - {self.name2}",
                opacity=0.7,
                nbinsx=20,
                histnorm='probability density'
            ))
        
        fig.update_layout(
            title=f"{metric} Distribution Comparison by {group_by}",
            xaxis_title=metric,
            yaxis_title="Density",
            barmode='overlay',
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
        view_mode = st.radio("Choose Mode", ["📊 Interactive Analysis", "📂 Dataset Explorer", "🔍 Advanced Metrics", "⚖️ Dataset Comparison"])
        
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
    
    elif view_mode == "⚖️ Dataset Comparison":
        st.markdown("### 📊 Dataset Comparison Analysis")
        st.markdown("*Compare two datasets with similar column structures*")
        
        # Dataset selection interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 First Dataset")
            dataset1 = st.selectbox("Select First Dataset", all_files, key="dataset1")
            
        with col2:
            st.markdown("#### 📄 Second Dataset")
            dataset2 = st.selectbox("Select Second Dataset", all_files, key="dataset2")
        
        if dataset1 and dataset2 and dataset1 != dataset2:
            # Load both datasets
            df1 = pd.read_csv(os.path.join(base_dir, dataset1))
            df2 = pd.read_csv(os.path.join(base_dir, dataset2))
            
            # Create comparator
            comparator = DatasetComparator(df1, df2, 
                                         os.path.basename(dataset1).replace('.csv', ''),
                                         os.path.basename(dataset2).replace('.csv', ''))
            
            # Check for common columns
            if not comparator.common_columns['all']:
                st.error("❌ No common columns found between datasets!")
                st.info("**Dataset 1 columns:** " + ", ".join(df1.columns.tolist()[:10]))
                st.info("**Dataset 2 columns:** " + ", ".join(df2.columns.tolist()[:10]))
                return
            
            # Display dataset info
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Common Columns", len(comparator.common_columns['all']))
            with info_col2:
                st.metric("Common Metrics", len(comparator.common_columns['numeric']))
            with info_col3:
                st.metric("Common Categories", len(comparator.common_columns['categorical']))
            
            # Comparison controls
            st.markdown("#### 🎛️ Comparison Controls")
            control_col1, control_col2, control_col3 = st.columns(3)
            
            with control_col1:
                comparison_metric = st.selectbox("Comparison Metric", comparator.common_columns['numeric'])
            
            with control_col2:
                group_by_col = st.selectbox("Group By", comparator.common_columns['categorical'])
            
            with control_col3:
                chart_style = st.selectbox("Chart Style", [
                    "📊 Side-by-Side Bars",
                    "📈 Performance Delta",
                    "📊 Distribution Overlay",
                    "🔥 Correlation Heatmap"
                ])
            
            # Generate comparison charts
            if comparison_metric and group_by_col:
                
                if chart_style == "📊 Side-by-Side Bars":
                    fig = comparator.create_comparison_bar_chart(comparison_metric, group_by_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics
                        st.markdown("#### 📊 Summary Statistics")
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.markdown(f"**{comparator.name1}**")
                            stats1 = df1.groupby(group_by_col)[comparison_metric].agg(['mean', 'std', 'min', 'max'])
                            st.dataframe(stats1.round(4))
                        
                        with summary_col2:
                            st.markdown(f"**{comparator.name2}**")
                            stats2 = df2.groupby(group_by_col)[comparison_metric].agg(['mean', 'std', 'min', 'max'])
                            st.dataframe(stats2.round(4))
                
                elif chart_style == "📈 Performance Delta":
                    fig = comparator.create_performance_delta_chart(comparison_metric, group_by_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show improvement/degradation summary
                        agg1 = df1.groupby(group_by_col)[comparison_metric].mean()
                        agg2 = df2.groupby(group_by_col)[comparison_metric].mean()
                        delta_df = pd.DataFrame({
                            'Dataset_1': agg1,
                            'Dataset_2': agg2,
                            'Delta': agg2 - agg1,
                            'Pct_Change': ((agg2 - agg1) / agg1 * 100).round(2)
                        }).reset_index()
                        
                        st.markdown("#### 📈 Performance Changes")
                        st.dataframe(delta_df, use_container_width=True)
                
                elif chart_style == "📊 Distribution Overlay":
                    fig = comparator.create_distribution_comparison(comparison_metric, group_by_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_style == "🔥 Correlation Heatmap":
                    fig = comparator.create_correlation_comparison()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            # Additional comparison insights
            st.markdown("#### 🔍 Quick Insights")
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("**Dataset Shapes:**")
                st.info(f"📄 {comparator.name1}: {df1.shape[0]} rows × {df1.shape[1]} cols")
                st.info(f"📄 {comparator.name2}: {df2.shape[0]} rows × {df2.shape[1]} cols")
            
            with insights_col2:
                if comparison_metric and group_by_col:
                    mean1 = df1[comparison_metric].mean()
                    mean2 = df2[comparison_metric].mean()
                    st.markdown("**Overall Performance:**")
                    st.metric(f"{comparator.name1} Avg", f"{mean1:.4f}")
                    st.metric(f"{comparator.name2} Avg", f"{mean2:.4f}", delta=f"{mean2-mean1:.4f}")
        
        else:
            st.info("👆 Please select two different datasets to compare")
    
    # Footer with dynamic info
    st.markdown("---")
    if 'analyzer' in locals():
        st.markdown(f"**Detected Structure:** {len(analyzer.metric_cols)} metrics, {len(analyzer.grouping_cols)} grouping columns, {len(analyzer.categorical_cols)} categorical features")
