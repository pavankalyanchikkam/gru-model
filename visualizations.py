"""
Visualization Generator for PrognosAI
Creates interactive Plotly charts for the dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class VisualizationGenerator:
    """Generate interactive visualizations for the dashboard"""
    
    @staticmethod
    def create_rul_chart(predictions, true_rul=None, title="RUL Predictions"):
        """Create interactive RUL comparison chart"""
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        fig = go.Figure()
        
        # Add predicted RUL
        fig.add_trace(go.Scatter(
            x=df['unit'],
            y=df['predicted_rul'],
            mode='lines+markers',
            name='Predicted RUL',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='Unit: %{x}<br>RUL: %{y:.1f} cycles<extra></extra>'
        ))
        
        # Add true RUL if available
        if true_rul is not None and len(true_rul) == len(predictions):
            fig.add_trace(go.Scatter(
                x=df['unit'],
                y=true_rul,
                mode='lines+markers',
                name='True RUL',
                line=dict(color='#10b981', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='Unit: %{x}<br>True RUL: %{y:.1f} cycles<extra></extra>'
            ))
        
        # Add alert zones
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='#1f2937')
            ),
            xaxis=dict(
                title='Engine Unit',
                gridcolor='#e5e7eb',
                showgrid=True
            ),
            yaxis=dict(
                title='Remaining Useful Life (cycles)',
                gridcolor='#e5e7eb',
                showgrid=True
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def create_alert_distribution(predictions, title="Alert Distribution"):
        """Create alert distribution visualization"""
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        alert_counts = df['alert'].value_counts().reset_index()
        alert_counts.columns = ['Status', 'Count']
        
        # Define colors for each status
        color_map = {
            'CRITICAL': '#dc2626',
            'WARNING': '#f59e0b',
            'NORMAL': '#10b981'
        }
        
        colors = [color_map.get(status, '#6b7280') for status in alert_counts['Status']]
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=alert_counts['Status'],
            values=alert_counts['Count'],
            hole=0.5,
            marker=dict(colors=colors),
            textinfo='label+percent+value',
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='#1f2937')
            ),
            height=450,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='right',
                x=1.2
            )
        )
        
        return fig
    
    @staticmethod
    def create_error_distribution(predictions, true_rul=None, title="Error Distribution"):
        """Create error distribution histogram"""
        if not predictions or true_rul is None:
            return None
        
        if len(predictions) != len(true_rul):
            return None
        
        pred_values = [p['predicted_rul'] for p in predictions]
        errors = np.array(pred_values) - np.array(true_rul)
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            name='Prediction Errors',
            marker_color='#3b82f6',
            opacity=0.7,
            hovertemplate='Error: %{x:.1f} cycles<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_error = np.mean(errors)
        fig.add_vline(
            x=mean_error,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Mean: {mean_error:.2f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='#1f2937')
            ),
            xaxis=dict(
                title='Prediction Error (cycles)',
                gridcolor='#e5e7eb',
                zeroline=True,
                zerolinecolor='#9ca3af',
                zerolinewidth=1
            ),
            yaxis=dict(
                title='Frequency',
                gridcolor='#e5e7eb'
            ),
            bargap=0.1,
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def create_trend_analysis(predictions, warning_thresh=50, critical_thresh=20):
        """Create trend analysis visualization"""
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        
        fig = go.Figure()
        
        # Add scatter plot with color by alert
        for alert_type in ['NORMAL', 'WARNING', 'CRITICAL']:
            alert_data = df[df['alert'] == alert_type]
            if len(alert_data) > 0:
                color = '#10b981' if alert_type == 'NORMAL' else \
                        '#f59e0b' if alert_type == 'WARNING' else '#dc2626'
                
                fig.add_trace(go.Scatter(
                    x=alert_data['unit'],
                    y=alert_data['predicted_rul'],
                    mode='markers',
                    name=alert_type,
                    marker=dict(
                        color=color,
                        size=10,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='Unit: %{x}<br>RUL: %{y:.1f} cycles<br>Status: ' + alert_type + '<extra></extra>'
                ))
        
        # Add threshold lines
        fig.add_hline(
            y=warning_thresh,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text=f"Warning: {warning_thresh} cycles",
            annotation_position="bottom right"
        )
        
        fig.add_hline(
            y=critical_thresh,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text=f"Critical: {critical_thresh} cycles",
            annotation_position="bottom right"
        )
        
        # Add trend line
        z = np.polyfit(df['unit'], df['predicted_rul'], 1)
        p = np.poly1d(z)
        trend_line = p(df['unit'])
        
        fig.add_trace(go.Scatter(
            x=df['unit'],
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color='#1f2937', width=2, dash='dash'),
            hovertemplate='Trend: %{y:.1f} cycles<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='RUL Trend with Alert Status',
                font=dict(size=18, color='#1f2937')
            ),
            xaxis=dict(
                title='Engine Unit',
                gridcolor='#e5e7eb'
            ),
            yaxis=dict(
                title='Remaining Useful Life (cycles)',
                gridcolor='#e5e7eb'
            ),
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_confidence_plot(predictions):
        """Create confidence level visualization"""
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        
        # Group by confidence ranges
        df['confidence_range'] = pd.cut(
            df['confidence'], 
            bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=['<60%', '60-70%', '70-80%', '80-90%', '>90%']
        )
        
        confidence_counts = df['confidence_range'].value_counts().sort_index().reset_index()
        confidence_counts.columns = ['Confidence', 'Count']
        
        fig = px.bar(
            confidence_counts,
            x='Confidence',
            y='Count',
            color='Confidence',
            color_discrete_sequence=['#dc2626', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981'],
            title='Prediction Confidence Distribution',
            labels={'Count': 'Number of Predictions', 'Confidence': 'Confidence Level'}
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis=dict(title='Confidence Level'),
            yaxis=dict(title='Count')
        )
        
        return fig