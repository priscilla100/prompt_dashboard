import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Example data - replace with your actual data
# Dataset 1 (Experiment 1)
data1 = {
    'Model': ['GPT-4', 'Claude-3', 'Gemini-Pro', 'Llama-2', 'Mistral-7B'],
    'Approach': ['Few-shot', 'Zero-shot', 'Chain-of-thought', 'Fine-tuned', 'RAG'],
    'Total_Samples': [1000, 1000, 1000, 1000, 1000],
    'Yes_Samples': [450, 420, 480, 390, 410],
    'No_Samples': [550, 580, 520, 610, 590],
    'True_Positives': [425, 385, 445, 360, 375],
    'True_Negatives': [520, 545, 485, 580, 555],
    'False_Positives': [30, 35, 35, 30, 35],
    'False_Negatives': [25, 35, 35, 30, 35],
    'Accuracy': [0.945, 0.930, 0.930, 0.940, 0.930],
    'Precision': [0.934, 0.917, 0.927, 0.923, 0.915],
    'Recall': [0.944, 0.917, 0.927, 0.923, 0.915],
    'F1_Score': [0.939, 0.917, 0.927, 0.923, 0.915]
}

# Dataset 2 (Experiment 2)
data2 = {
    'Model': ['GPT-4', 'Claude-3', 'Gemini-Pro', 'Llama-2', 'Mistral-7B'],
    'Approach': ['Few-shot-Rich', 'Zero-shot-Rich', 'CoT-Rich', 'Fine-tuned-Rich', 'RAG-Rich'],
    'Total_Samples': [1000, 1000, 1000, 1000, 1000],
    'Yes_Samples': [460, 435, 490, 405, 425],
    'No_Samples': [540, 565, 510, 595, 575],
    'True_Positives': [445, 415, 470, 385, 405],
    'True_Negatives': [525, 550, 495, 585, 565],
    'False_Positives': [15, 15, 15, 10, 10],
    'False_Negatives': [15, 20, 20, 20, 20],
    'Accuracy': [0.970, 0.965, 0.965, 0.970, 0.970],
    'Precision': [0.967, 0.965, 0.969, 0.975, 0.976],
    'Recall': [0.967, 0.954, 0.959, 0.951, 0.953],
    'F1_Score': [0.967, 0.959, 0.964, 0.963, 0.964]
}
# /Users/priscilladanso/Documents/GitHub/prompt_dashboard/data/undefined/wff_aggregate_metrics.csv
df1 = pd.read_csv("/Users/priscilladanso/Documents/GitHub/prompt_dashboard/data/undefined/wff_aggregate_metrics.csv")
df2 = pd.read_csv("/Users/priscilladanso/Documents/GitHub/prompt_dashboard/data/defined/wff_aggregate_metrics.csv")

# Add experiment labels
df1['Experiment'] = 'Standard Prompts'
df2['Experiment'] = 'Rich Prompts'

# Combine datasets
df_combined = pd.concat([df1, df2], ignore_index=True)

class ModelComparisonAnalyzer:
    def __init__(self, df_combined):
        self.df = df_combined
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A2B']
        self.paper_style = {
            'font_family': 'Times New Roman',
            'font_size': 14,
            'title_font_size': 18,
            'legend_font_size': 12
        }
    
    def create_comparison_table(self):
        """Create a professional comparison table"""
        # Calculate improvements
        comparison_data = []
        
        for model in df1['Model'].unique():
            row1 = df1[df1['Model'] == model].iloc[0]
            row2 = df2[df2['Model'] == model].iloc[0]
            
            improvement = {
                'Model': model,
                'Approach': row1['Approach'] + ' → ' + row2['Approach'],
                'Accuracy_Std': f"{row1['Accuracy']:.3f}",
                'Accuracy_Rich': f"{row2['Accuracy']:.3f}",
                'Accuracy_Δ': f"{(row2['Accuracy'] - row1['Accuracy']):.3f}",
                'Precision_Std': f"{row1['Precision']:.3f}",
                'Precision_Rich': f"{row2['Precision']:.3f}",
                'Precision_Δ': f"{(row2['Precision'] - row1['Precision']):.3f}",
                'F1_Std': f"{row1['F1_Score']:.3f}",
                'F1_Rich': f"{row2['F1_Score']:.3f}",
                'F1_Δ': f"{(row2['F1_Score'] - row1['F1_Score']):.3f}"
            }
            comparison_data.append(improvement)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create table visualization
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Model</b>', '<b>Approach</b>', 
                       '<b>Accuracy</b><br>Standard', '<b>Accuracy</b><br>Rich', '<b>Δ Accuracy</b>',
                       '<b>Precision</b><br>Standard', '<b>Precision</b><br>Rich', '<b>Δ Precision</b>',
                       '<b>F1 Score</b><br>Standard', '<b>F1 Score</b><br>Rich', '<b>Δ F1</b>'],
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='black'),
                height=40
            ),
            cells=dict(
                values=[comparison_df[col] for col in comparison_df.columns],
                fill_color='white',
                align='center',
                font=dict(size=11),
                height=35
            )
        )])
        
        fig.update_layout(
            title="Model Performance Comparison: Standard vs Rich Prompts",
            font=dict(family="Times New Roman", size=14),
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def create_accuracy_comparison_chart(self):
        """Create a professional bar chart comparing accuracy"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy by Model', 'Precision vs Recall'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy comparison
        models = df1['Model'].unique()
        std_accuracy = [df1[df1['Model'] == m]['Accuracy'].iloc[0] for m in models]
        rich_accuracy = [df2[df2['Model'] == m]['Accuracy'].iloc[0] for m in models]
        
        fig.add_trace(
            go.Bar(
                name='Standard Prompts',
                x=models,
                y=std_accuracy,
                marker_color='#2E86AB',
                text=[f'{val:.3f}' for val in std_accuracy],
                textposition='auto',
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Rich Prompts',
                x=models,
                y=rich_accuracy,
                marker_color='#A23B72',
                text=[f'{val:.3f}' for val in rich_accuracy],
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(
                x=df1['Precision'],
                y=df1['Recall'],
                mode='markers+text',
                name='Standard Prompts',
                text=df1['Model'],
                textposition='top center',
                marker=dict(size=12, color='#2E86AB'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=df2['Precision'],
                y=df2['Recall'],
                mode='markers+text',
                name='Rich Prompts',
                text=df2['Model'],
                textposition='bottom center',
                marker=dict(size=12, color='#A23B72'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Model Performance Analysis",
            font=dict(family="Times New Roman", size=14),
            height=500,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Precision", row=1, col=2)
        fig.update_yaxes(title_text="Recall", row=1, col=2)
        
        return fig
    
    def create_error_analysis_chart(self):
        """Create error analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('False Positives', 'False Negatives', 
                          'Error Rate Distribution', 'Performance Metrics Heatmap'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        models = df1['Model'].unique()
        
        # False Positives
        fp_std = [df1[df1['Model'] == m]['False_Positives'].iloc[0] for m in models]
        fp_rich = [df2[df2['Model'] == m]['False_Positives'].iloc[0] for m in models]
        
        fig.add_trace(go.Bar(x=models, y=fp_std, name='Standard', marker_color='#2E86AB'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=fp_rich, name='Rich', marker_color='#A23B72'), row=1, col=1)
        
        # False Negatives
        fn_std = [df1[df1['Model'] == m]['False_Negatives'].iloc[0] for m in models]
        fn_rich = [df2[df2['Model'] == m]['False_Negatives'].iloc[0] for m in models]
        
        fig.add_trace(go.Bar(x=models, y=fn_std, name='Standard', marker_color='#2E86AB', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=models, y=fn_rich, name='Rich', marker_color='#A23B72', showlegend=False), row=1, col=2)
        
        # Error Rate Distribution
        error_rate_std = [(fp + fn) / 1000 for fp, fn in zip(fp_std, fn_std)]
        error_rate_rich = [(fp + fn) / 1000 for fp, fn in zip(fp_rich, fn_rich)]
        
        fig.add_trace(go.Scatter(
            x=models, y=error_rate_std, mode='markers+lines',
            name='Standard Error Rate', marker_color='#2E86AB'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=models, y=error_rate_rich, mode='markers+lines',
            name='Rich Error Rate', marker_color='#A23B72'
        ), row=2, col=1)
        
        # Performance Metrics Heatmap
        metrics_data = []
        for i, model in enumerate(models):
            std_row = df1[df1['Model'] == model].iloc[0]
            rich_row = df2[df2['Model'] == model].iloc[0]
            
            metrics_data.extend([
                [model + ' (Std)', 'Accuracy', std_row['Accuracy']],
                [model + ' (Std)', 'Precision', std_row['Precision']],
                [model + ' (Std)', 'Recall', std_row['Recall']],
                [model + ' (Rich)', 'Accuracy', rich_row['Accuracy']],
                [model + ' (Rich)', 'Precision', rich_row['Precision']],
                [model + ' (Rich)', 'Recall', rich_row['Recall']]
            ])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'Metric', 'Value'])
        pivot_df = metrics_df.pivot(index='Model', columns='Metric', values='Value')
        
        fig.add_trace(go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='Viridis',
            text=np.round(pivot_df.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True
        ), row=2, col=2)
        
        fig.update_layout(
            title="Comprehensive Error Analysis",
            font=dict(family="Times New Roman", size=14),
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_improvement_analysis(self):
        """Create improvement analysis chart"""
        improvements = []
        models = df1['Model'].unique()
        
        for model in models:
            std_row = df1[df1['Model'] == model].iloc[0]
            rich_row = df2[df2['Model'] == model].iloc[0]
            
            improvements.append({
                'Model': model,
                'Accuracy_Improvement': rich_row['Accuracy'] - std_row['Accuracy'],
                'Precision_Improvement': rich_row['Precision'] - std_row['Precision'],
                'Recall_Improvement': rich_row['Recall'] - std_row['Recall'],
                'F1_Improvement': rich_row['F1_Score'] - std_row['F1_Score']
            })
        
        imp_df = pd.DataFrame(improvements)
        
        fig = go.Figure()
        
        metrics = ['Accuracy_Improvement', 'Precision_Improvement', 'Recall_Improvement', 'F1_Improvement']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric.replace('_Improvement', ''),
                x=imp_df['Model'],
                y=imp_df[metric],
                marker_color=colors[i],
                text=[f'{val:.3f}' for val in imp_df[metric]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Performance Improvements: Rich vs Standard Prompts",
            xaxis_title="Models",
            yaxis_title="Improvement (Rich - Standard)",
            font=dict(family="Times New Roman", size=14),
            height=500,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        return fig
    
    def generate_insights(self):
        """Generate insights about the comparison"""
        insights = []
        
        # Overall improvement analysis
        models = df1['Model'].unique()
        improvements = []
        
        for model in models:
            std_row = df1[df1['Model'] == model].iloc[0]
            rich_row = df2[df2['Model'] == model].iloc[0]
            
            improvements.append({
                'model': model,
                'acc_imp': rich_row['Accuracy'] - std_row['Accuracy'],
                'prec_imp': rich_row['Precision'] - std_row['Precision'],
                'rec_imp': rich_row['Recall'] - std_row['Recall'],
                'f1_imp': rich_row['F1_Score'] - std_row['F1_Score']
            })
        
        # Calculate average improvements
        avg_acc_imp = np.mean([imp['acc_imp'] for imp in improvements])
        avg_prec_imp = np.mean([imp['prec_imp'] for imp in improvements])
        avg_f1_imp = np.mean([imp['f1_imp'] for imp in improvements])
        
        insights.append(f"📊 **Overall Impact**: Rich prompts show an average improvement of {avg_acc_imp:.3f} in accuracy, {avg_prec_imp:.3f} in precision, and {avg_f1_imp:.3f} in F1-score.")
        
        # Best performing model
        best_model_std = df1.loc[df1['Accuracy'].idxmax(), 'Model']
        best_model_rich = df2.loc[df2['Accuracy'].idxmax(), 'Model']
        
        insights.append(f"🏆 **Best Performers**: {best_model_std} leads with standard prompts, while {best_model_rich} excels with rich prompts.")
        
        # Biggest improvement
        best_improvement = max(improvements, key=lambda x: x['acc_imp'])
        insights.append(f"📈 **Biggest Improvement**: {best_improvement['model']} shows the largest accuracy gain of {best_improvement['acc_imp']:.3f}.")
        
        # Error reduction
        total_errors_std = df1['False_Positives'].sum() + df1['False_Negatives'].sum()
        total_errors_rich = df2['False_Positives'].sum() + df2['False_Negatives'].sum()
        error_reduction = (total_errors_std - total_errors_rich) / total_errors_std
        
        insights.append(f"🎯 **Error Reduction**: Rich prompts reduce total errors by {error_reduction:.1%} across all models.")
        
        return insights

# Initialize analyzer
analyzer = ModelComparisonAnalyzer(df_combined)

# Generate all visualizations
print("=== PROFESSIONAL MODEL COMPARISON ANALYSIS ===\n")

# Display insights
insights = analyzer.generate_insights()
print("KEY INSIGHTS:")
for insight in insights:
    print(insight)

print("\n" + "="*60)
print("GENERATING PUBLICATION-READY CHARTS...")
print("="*60)

# Create all charts
table_fig = analyzer.create_comparison_table()
accuracy_fig = analyzer.create_accuracy_comparison_chart()
error_fig = analyzer.create_error_analysis_chart()
improvement_fig = analyzer.create_improvement_analysis()

# Display charts
table_fig.show()
accuracy_fig.show()
error_fig.show()
improvement_fig.show()
table_fig.write_image('dashboard/wff/comparison_table.png', width=1200, height=400, scale=2)
accuracy_fig.write_image('dashboard/wff/accuracy_comparison.png', width=1200, height=500, scale=2)
error_fig.write_image('dashboard/wff/error_analysis.png', width=1200, height=800, scale=2)
improvement_fig.write_image('dashboard/wff/improvement_analysis.png', width=1200, height=500, scale=2)
print("\n📊 All charts generated successfully!")
print("💡 To save charts for your paper:")
print("   - table_fig.write_image('comparison_table.png', width=1200, height=400, scale=2)")
print("   - accuracy_fig.write_image('accuracy_comparison.png', width=1200, height=500, scale=2)")
print("   - error_fig.write_image('error_analysis.png', width=1200, height=800, scale=2)")
print("   - improvement_fig.write_image('improvement_analysis.png', width=1200, height=500, scale=2)")