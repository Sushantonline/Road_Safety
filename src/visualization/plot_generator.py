import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import numpy as np

class PlotGenerator:
    def __init__(self):
        self.plot_types = {
            'trend': self._create_trend_plot,
            'comparison': self._create_comparison_plot,
            'distribution': self._create_distribution_plot,
            'correlation': self._create_correlation_plot,
            'ranking': self._create_ranking_plot,
            'geographical': self._create_geo_plot
        }
    
    def recommend_plots(self, query: str, data: pd.DataFrame, intent_info: Dict) -> List[str]:
        """Recommend appropriate plot types based on query and data"""
        recommendations = []
        
        # Check data characteristics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Check for time series data
        has_year = any('year' in col.lower() for col in data.columns)
        has_time = any('time' in col.lower() or 'date' in col.lower() for col in data.columns)
        
        # Recommend based on intent
        if 'trend_analysis' in intent_info.get('intents', []) and (has_year or has_time):
            recommendations.append('trend')
        
        if 'comparison' in intent_info.get('intents', []):
            recommendations.append('comparison')
        
        if 'ranking' in intent_info.get('intents', []):
            recommendations.append('ranking')
        
        if len(numeric_cols) >= 2:
            recommendations.append('correlation')
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append('distribution')
        
        # Check for geographical data
        if any('state' in col.lower() for col in data.columns):
            recommendations.append('geographical')
        
        return recommendations if recommendations else ['distribution']
    
    def generate_plot(self, plot_type: str, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Generate specific plot type"""
        if plot_type in self.plot_types:
            return self.plot_types[plot_type](data, **kwargs)
        else:
            return self._create_default_plot(data)
    
    def _create_trend_plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create trend/time series plot"""
        # Find year/time column
        time_col = None
        for col in data.columns:
            if 'year' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col is None:
            return self._create_default_plot(data)
        
        # Find numeric columns for plotting
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != time_col]
        
        fig = go.Figure()
        
        for col in numeric_cols[:3]:  # Limit to 3 series for readability
            fig.add_trace(go.Scatter(
                x=data[time_col],
                y=data[col],
                mode='lines+markers',
                name=col.replace('_', ' ').title(),
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title=f"Trend Analysis Over Time",
            xaxis_title=time_col.replace('_', ' ').title(),
            yaxis_title="Values",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_comparison_plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create comparison plot (bar chart)"""
        # Find categorical and numeric columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return self._create_default_plot(data)
        
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Aggregate data if needed
        if len(data) > 20:
            plot_data = data.groupby(cat_col)[num_col].sum().head(15).reset_index()
        else:
            plot_data = data.head(15)
        
        fig = px.bar(
            plot_data,
            x=cat_col,
            y=num_col,
            title=f"Comparison of {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
            template='plotly_white'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig
    
    def _create_ranking_plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create ranking plot (horizontal bar chart)"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return self._create_default_plot(data)
        
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Sort and take top 15
        plot_data = data.sort_values(num_col, ascending=False).head(15)
        
        fig = px.bar(
            plot_data,
            y=cat_col,
            x=num_col,
            orientation='h',
            title=f"Top Rankings: {num_col.replace('_', ' ').title()}",
            template='plotly_white'
        )
        
        return fig
    
    def _create_distribution_plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create distribution plot"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return self._create_default_plot(data)
        
        col = numeric_cols[0]
        
        fig = px.histogram(
            data,
            x=col,
            title=f"Distribution of {col.replace('_', ' ').title()}",
            template='plotly_white',
            nbins=30
        )
        
        return fig
    
    def _create_correlation_plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create correlation heatmap"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return self._create_default_plot(data)
        
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto',
            template='plotly_white'
        )
        
        return fig
    
    def _create_geo_plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create geographical plot for Indian states"""
        # Check if we have state data
        state_col = None
        for col in data.columns:
            if 'state' in col.lower():
                state_col = col
                break
        
        if state_col is None:
            return self._create_default_plot(data)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return self._create_default_plot(data)
        
        value_col = numeric_cols[0]
        
        # Create a choropleth map (simplified version)
        fig = px.bar(
            data.head(20),
            x=state_col,
            y=value_col,
            title=f"{value_col.replace('_', ' ').title()} by State",
            template='plotly_white'
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig
    
    def _create_default_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create default plot when specific plot type can't be determined"""
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Simple scatter plot of first two numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            fig = px.scatter(
                data.head(100),
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                template='plotly_white'
            )
        else:
            # Bar plot of first numeric column
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.bar(
                    data.head(20),
                    y=col,
                    title=f"Distribution of {col}",
                    template='plotly_white'
                )
            else:
                # Just show data info
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Data shape: {data.shape}\nColumns: {', '.join(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=12)
                )
        
        return fig

class PlotRecommender:
    def __init__(self, plot_generator: PlotGenerator):
        self.plot_generator = plot_generator
    
    def get_recommended_plots(self, query: str, data: pd.DataFrame, intent_info: Dict) -> List[Dict[str, Any]]:
        """Get recommended plots with generated figures"""
        plot_types = self.plot_generator.recommend_plots(query, data, intent_info)
        
        recommended_plots = []
        for plot_type in plot_types:
            try:
                fig = self.plot_generator.generate_plot(plot_type, data)
                recommended_plots.append({
                    'type': plot_type,
                    'figure': fig,
                    'title': f"{plot_type.replace('_', ' ').title()} Plot",
                    'description': self._get_plot_description(plot_type)
                })
            except Exception as e:
                print(f"Error generating {plot_type} plot: {e}")
                continue
        
        return recommended_plots
    
    def _get_plot_description(self, plot_type: str) -> str:
        """Get description for plot type"""
        descriptions = {
            'trend': "Shows how values change over time",
            'comparison': "Compares values across different categories",
            'ranking': "Shows top performers in descending order",
            'distribution': "Shows the frequency distribution of values",
            'correlation': "Shows relationships between different variables",
            'geographical': "Shows data distribution across geographical regions"
        }
        return descriptions.get(plot_type, "Visualizes the data")
