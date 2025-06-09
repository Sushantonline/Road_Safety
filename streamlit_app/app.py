import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm.query_processor import QueryProcessor, ResponseGenerator
from visualization.plot_generator import PlotGenerator, PlotRecommender
from data_processing.text_processor import TextProcessor, COTDatasetGenerator
from utils.config import config

# Configure Streamlit page
st.set_page_config(
    page_title="Road Accident Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class RoadAccidentApp:
    def __init__(self):
        self.init_session_state()
        self.load_models()
        self.load_sample_data()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'dataframes' not in st.session_state:
            st.session_state.dataframes = {}
        if 'current_data' not in st.session_state:
            st.session_state.current_data = pd.DataFrame()
    
    def load_models(self):
        """Load ML models and processors"""
        try:
            self.query_processor = QueryProcessor()
            self.response_generator = ResponseGenerator(self.query_processor)
            self.plot_generator = PlotGenerator()
            self.plot_recommender = PlotRecommender(self.plot_generator)
            self.text_processor = TextProcessor()
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Create sample dataframes based on the Indian road accident data
        
        # Yearly statistics
        yearly_data = {
            'year': [2018, 2019, 2020, 2021, 2022],
            'accidents': [470403, 456959, 372181, 412432, 461312],
            'fatalities': [157593, 158984, 138383, 153972, 168491],
            'injuries': [464715, 449360, 346747, 384448, 443366],
            'fatal_accidents': [143738, 145332, 127307, 142163, 155781]
        }
        
        # State-wise data (sample)
        state_data = {
            'state': ['Tamil Nadu', 'Maharashtra', 'Uttar Pradesh', 'Karnataka', 'Kerala', 
                     'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'Andhra Pradesh', 'Bihar'],
            'accidents': [64105, 54432, 45500, 40000, 38000, 35000, 32000, 30000, 28000, 25000],
            'fatalities': [17884, 13584, 22595, 11300, 9500, 8500, 9800, 8200, 7500, 6800],
            'population': [77841267, 123144223, 241066874, 67562686, 35699443, 
                          70090271, 78230816, 85358965, 53903393, 124799926]
        }
        
        # Road type data
        road_type_data = {
            'road_type': ['National Highways', 'State Highways', 'Other Roads'],
            'accidents': [151997, 106682, 202633],
            'fatalities': [61038, 41012, 66441],
            'road_length_km': [132499, 179535, 6019723],
            'percentage_of_network': [2.1, 2.8, 95.1]
        }
        
        # Cause of accidents
        cause_data = {
            'cause': ['Over-speeding', 'Driving on wrong side', 'Drunken driving', 
                     'Jumping red light', 'Use of mobile phone', 'Others'],
            'accidents': [333323, 22586, 10080, 4021, 7558, 83744],
            'fatalities': [119904, 9094, 4201, 1462, 3395, 30435],
            'percentage': [72.3, 4.9, 2.2, 0.9, 1.6, 18.1]
        }
        
        st.session_state.dataframes = {
            'yearly_stats': pd.DataFrame(yearly_data),
            'state_wise': pd.DataFrame(state_data),
            'road_type': pd.DataFrame(road_type_data),
            'accident_causes': pd.DataFrame(cause_data)
        }
        
        # Load dataframes into query processor
        self.query_processor.load_dataframes(st.session_state.dataframes)
    
    def render_sidebar(self):
        """Render sidebar with navigation and options"""
        st.sidebar.title("🚗 Road Accident Analytics")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["🏠 Main Dashboard", "📊 Data Explorer", "🤖 Query Interface", "📈 Custom Analysis"]
        )
        
        # Model settings
        st.sidebar.subheader("Model Settings")
        prompt_type = st.sidebar.selectbox(
            "Prompt Type:",
            ["analysis", "cot", "comparison"],
            help="Choose the type of reasoning approach"
        )
        
        # Data selection
        st.sidebar.subheader("Data Selection")
        available_datasets = list(st.session_state.dataframes.keys())
        selected_dataset = st.sidebar.selectbox(
            "Select Dataset:",
            available_datasets
        )
        
        return page, prompt_type, selected_dataset
    
    def render_main_dashboard(self):
        """Render main dashboard"""
        st.markdown('<h1 class="main-header">🚗 Indian Road Accident Analysis System</h1>', 
                   unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        yearly_stats = st.session_state.dataframes['yearly_stats']
        latest_year = yearly_stats.iloc[-1]
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Accidents (2022)", f"{latest_year['accidents']:,}", 
                     f"{latest_year['accidents'] - yearly_stats.iloc[-2]['accidents']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Fatalities (2022)", f"{latest_year['fatalities']:,}", 
                     f"{latest_year['fatalities'] - yearly_stats.iloc[-2]['fatalities']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Injuries (2022)", f"{latest_year['injuries']:,}", 
                     f"{latest_year['injuries'] - yearly_stats.iloc[-2]['injuries']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Fatal Accidents (2022)", f"{latest_year['fatal_accidents']:,}", 
                     f"{latest_year['fatal_accidents'] - yearly_stats.iloc[-2]['fatal_accidents']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Trend visualization
        st.markdown('<h2 class="sub-header">📈 Trends Over Time</h2>', unsafe_allow_html=True)
        
        fig = px.line(yearly_stats, x='year', y=['accidents', 'fatalities', 'injuries'],
                     title="Road Accident Trends (2018-2022)",
                     labels={'value': 'Count', 'variable': 'Metric'})
        st.plotly_chart(fig, use_container_width=True)
        
        # State-wise analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">🗺️ Top States by Accidents</h3>', unsafe_allow_html=True)
            state_data = st.session_state.dataframes['state_wise'].head(10)
            fig = px.bar(state_data, x='accidents', y='state', orientation='h',
                        title="Accidents by State")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="sub-header">💀 Top States by Fatalities</h3>', unsafe_allow_html=True)
            fig = px.bar(state_data, x='fatalities', y='state', orientation='h',
                        title="Fatalities by State")
            st.plotly_chart(fig, use_container_width=True)
        
        # Accident causes
        st.markdown('<h3 class="sub-header">🚨 Major Causes of Accidents</h3>', unsafe_allow_html=True)
        cause_data = st.session_state.dataframes['accident_causes']
        fig = px.pie(cause_data, values='accidents', names='cause',
                    title="Distribution of Accident Causes")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_query_interface(self, prompt_type):
        """Render natural language query interface"""
        st.markdown('<h1 class="main-header">🤖 Natural Language Query Interface</h1>', 
                   unsafe_allow_html=True)
        
        # Query input
        st.markdown('<h3 class="sub-header">Ask a Question</h3>', unsafe_allow_html=True)
        
        # Sample queries
        sample_queries = [
            "What is the trend in road accidents over the years?",
            "Compare accident rates between Tamil Nadu and Maharashtra",
            "What are the top 5 states with highest fatalities?",
            "What percentage of accidents are caused by over-speeding?",
            "Show me the correlation between accidents and fatalities",
            "Which road type has the highest accident rate per km?"
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What is the trend in road accidents over the past 5 years?",
                height=100
            )
        
        with col2:
            st.write("**Sample Queries:**")
            for i, sample in enumerate(sample_queries[:3]):
                if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                    query = sample
                    st.experimental_rerun()
        
        # Query processing
        if st.button("🔍 Analyze Query", type="primary") and query:
            with st.spinner("Processing your query..."):
                try:
                    # Generate response
                    result = self.response_generator.generate_response(query, prompt_type)
                    
                    # Display response
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown(f"**Analysis Result:**\n\n{result['response']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display data used
                    if not result['data_used'].empty:
                        st.markdown('<h3 class="sub-header">📊 Data Used</h3>', unsafe_allow_html=True)
                        st.dataframe(result['data_used'])
                    
                    # Generate visualizations if needed
                    if result['needs_visualization'] and not result['data_used'].empty:
                        st.markdown('<h3 class="sub-header">📈 Recommended Visualizations</h3>', 
                                   unsafe_allow_html=True)
                        
                        plots = self.plot_recommender.get_recommended_plots(
                            query, result['data_used'], result['intent_info']
                        )
                        
                        for plot in plots:
                            st.markdown(f"**{plot['title']}**")
                            st.plotly_chart(plot['figure'], use_container_width=True)
                            st.caption(plot['description'])
                    
                    # Add to history
                    st.session_state.query_history.append({
                        'query': query,
                        'prompt_type': prompt_type,
                        'response': result['response']
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
        
        # Query history
        if st.session_state.query_history:
            st.markdown('<h3 class="sub-header">📝 Query History</h3>', unsafe_allow_html=True)
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {item['query'][:50]}..."):
                    st.write(f"**Prompt Type:** {item['prompt_type']}")
                    st.write(f"**Response:** {item['response']}")
    
    def render_data_explorer(self, selected_dataset):
        """Render data explorer"""
        st.markdown('<h1 class="main-header">📊 Data Explorer</h1>', unsafe_allow_html=True)
        
        if selected_dataset in st.session_state.dataframes:
            df = st.session_state.dataframes[selected_dataset]
            
            # Dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Data Types", df.dtypes.nunique())
            
            # Data preview
            st.markdown('<h3 class="sub-header">🔍 Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            
            # Data summary
            st.markdown('<h3 class="sub-header">📈 Statistical Summary</h3>', unsafe_allow_html=True)
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            
            # Quick visualizations
            st.markdown('<h3 class="sub-header">📊 Quick Visualizations</h3>', unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("X-axis", numeric_cols, key="x_axis")
                        y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key="y_axis")
                        
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        cat_col = st.selectbox("Category", categorical_cols, key="cat_col")
                        num_col = st.selectbox("Value", numeric_cols, key="num_col")
                        
                        fig = px.bar(df.head(10), x=cat_col, y=num_col, 
                                   title=f"{num_col} by {cat_col}")
                        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the Streamlit app"""
        page, prompt_type, selected_dataset = self.render_sidebar()
        
        if page == "🏠 Main Dashboard":
            self.render_main_dashboard()
        elif page == "📊 Data Explorer":
            self.render_data_explorer(selected_dataset)
        elif page == "🤖 Query Interface":
            self.render_query_interface(prompt_type)
        elif page == "📈 Custom Analysis":
            st.markdown('<h1 class="main-header">📈 Custom Analysis</h1>', unsafe_allow_html=True)
            st.info("Custom analysis tools coming soon!")

if __name__ == "__main__":
    app = RoadAccidentApp()
    app.run()
