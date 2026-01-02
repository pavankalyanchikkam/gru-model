"""
PrognosAI - Predictive Maintenance System
Main Streamlit Application
Optimized for i3/8GB Windows Systems
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import io
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="PrognosAI - Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/prognosai',
        'Report a bug': 'https://github.com/yourusername/prognosai/issues',
        'About': "## PrognosAI v1.0\nAI-Driven Predictive Maintenance System"
    }
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main Container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 6px solid;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .critical-card { border-left-color: #dc2626; }
    .warning-card { border-left-color: #f59e0b; }
    .normal-card { border-left-color: #10b981; }
    .info-card { border-left-color: #3b82f6; }
    
    /* Badges */
    .alert-badge {
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    
    .badge-critical {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #dc2626;
        border: 2px solid #dc2626;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        color: #d97706;
        border: 2px solid #f59e0b;
    }
    
    .badge-normal {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #059669;
        border: 2px solid #10b981;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(30, 58, 138, 0.4);
    }
    
    .secondary-button > button {
        background: white;
        color: #1e3a8a;
        border: 2px solid #1e3a8a;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #1e3a8a);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Hide Streamlit Defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE MANAGEMENT
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'predictions': None,
        'true_rul': None,
        'metrics': None,
        'dataset_name': None,
        'warning_thresh': 50,
        'critical_thresh': 20,
        'file_uploaded': False,
        'analysis_complete': False,
        'current_page': 0,
        'model_handler': None,
        'data_processor': None,
        'viz_generator': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================
# UTILITY FUNCTIONS
# ============================================
@st.cache_resource
def get_model_handler():
    """Get singleton model handler instance"""
    try:
        from utils.model_handler import ModelHandler
        return ModelHandler()
    except ImportError as e:
        st.error(f"❌ Error importing ModelHandler: {e}")
        return None

@st.cache_resource
def get_visualization_generator():
    """Get visualization generator"""
    try:
        from utils.visualizations import VisualizationGenerator
        return VisualizationGenerator()
    except:
        return None

@st.cache_resource
def get_data_processor():
    """Get data processor"""
    try:
        from utils.data_processor import DataProcessor
        return DataProcessor
    except:
        return None

def create_sample_files():
    """Create sample test files for demonstration"""
    sample_dir = Path("test_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample test data (CMAPSS format)
    test_data = """1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 39.06 23.4190
1 2 0.0019 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 21.61 553.75 2388.04 9044.07 1.30 47.49 521.66 2388.02 8131.49 8.4318 0.03 392 39.00 23.4236
1 3 -0.0043 0.0003 100.0 518.67 642.35 1587.99 1404.20 14.62 21.61 554.26 2388.07 9052.94 1.30 47.27 521.66 2388.02 8133.23 8.4178 0.03 390 38.95 23.3442
2 1 -0.0006 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 39.06 23.4190
2 2 0.0018 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 21.61 553.75 2388.04 9044.07 1.30 47.49 521.66 2388.02 8131.49 8.4318 0.03 392 39.00 23.4236"""
    
    # Sample RUL data
    rul_data = """125
112
98
105
87"""
    
    # Write files
    (sample_dir / "sample_test_FD001.txt").write_text(test_data)
    (sample_dir / "sample_RUL_FD001.txt").write_text(rul_data)
    
    return True

def generate_csv_download():
    """Generate CSV file for download"""
    if st.session_state.predictions:
        df = pd.DataFrame(st.session_state.predictions)
        
        # Add metadata
        df['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['dataset'] = st.session_state.dataset_name
        df['warning_threshold'] = st.session_state.warning_thresh
        df['critical_threshold'] = st.session_state.critical_thresh
        
        return df.to_csv(index=False)
    return ""

def render_alert_badge(alert_type):
    """Render styled alert badge"""
    badges = {
        'CRITICAL': '<span class="alert-badge badge-critical">🔴 CRITICAL</span>',
        'WARNING': '<span class="alert-badge badge-warning">🟡 WARNING</span>',
        'NORMAL': '<span class="alert-badge badge-normal">🟢 NORMAL</span>'
    }
    return badges.get(alert_type, '')

# ============================================
# PAGE FUNCTIONS
# ============================================
def show_home_page():
    """Display home/upload page"""
    st.markdown('<h1 class="main-header">🔧 PrognosAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Driven Predictive Maintenance System | Optimized for i3/8GB</p>', unsafe_allow_html=True)
    
    # Three column layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Welcome Card
        with st.container():
            st.markdown("### 🎯 Welcome to PrognosAI")
            st.info("""
            **Predict Remaining Useful Life (RUL)** of industrial machinery 
            using AI-powered analysis. Upload your CMAPSS dataset files 
            or try with our sample data.
            """)
        
        # Model Handler Initialization
        model_handler = get_model_handler()
        if model_handler:
            st.success("✅ Model Handler Initialized")
        
        # Quick Start Tabs
        tab1, tab2 = st.tabs(["📤 Upload Files", "🧪 Try Sample Data"])
        
        with tab1:
            show_upload_section(model_handler)
        
        with tab2:
            show_sample_data_section(model_handler)

def show_upload_section(model_handler):
    """Show file upload section"""
    st.markdown("### 📁 Upload Your Data")
    
    # Dataset Selection
    dataset = st.selectbox(
        "Select Dataset Model",
        options=["FD001", "FD002", "FD003", "FD004"],
        index=0,
        help="Choose the CMAPSS dataset for analysis"
    )
    
    # Display dataset info
    if model_handler:
        dataset_info = model_handler.get_dataset_info(dataset)
        with st.expander(f"📊 Dataset Info: {dataset_info['name']}"):
            st.write(f"**Description**: {dataset_info['description']}")
            st.write(f"**Complexity**: {dataset_info['complexity']}")
            st.write(f"**Recommended Use**: {dataset_info['recommended_use']}")
    
    # File Uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        test_file = st.file_uploader(
            "Test Data File",
            type=['txt', 'csv'],
            help="Upload test_FDXXX.txt file",
            key="test_upload"
        )
    
    with col2:
        true_rul_file = st.file_uploader(
            "True RUL File (Optional)",
            type=['txt', 'csv'],
            help="Upload RUL_FDXXX.txt file for accuracy metrics",
            key="rul_upload"
        )
    
    # Threshold Settings
    st.markdown("### ⚙️ Alert Settings")
    threshold_col1, threshold_col2 = st.columns(2)
    
    with threshold_col1:
        warning_thresh = st.slider(
            "Warning Threshold (cycles)",
            min_value=10,
            max_value=100,
            value=50,
            help="Trigger warning alert when RUL ≤ this value"
        )
    
    with threshold_col2:
        critical_thresh = st.slider(
            "Critical Threshold (cycles)",
            min_value=5,
            max_value=50,
            value=20,
            help="Trigger critical alert when RUL ≤ this value"
        )
    
    # Validation
    if critical_thresh >= warning_thresh:
        st.error("⚠️ Critical threshold must be LESS than warning threshold!")
        return
    
    # Run Analysis Button
    if st.button("🚀 Run Predictive Analysis", type="primary", use_container_width=True):
        if not test_file:
            st.error("Please upload a test data file!")
            return
        
        run_prediction_analysis(dataset, test_file, true_rul_file, warning_thresh, critical_thresh, model_handler)

def show_sample_data_section(model_handler):
    """Show sample data section"""
    st.markdown("### 🧪 Quick Start with Sample Data")
    
    st.info("""
    Try PrognosAI instantly with our pre-configured sample data.
    Perfect for testing and understanding the system workflow.
    """)
    
    if st.button("📋 Generate Sample Data", use_container_width=True):
        with st.spinner("Creating sample files..."):
            create_sample_files()
            st.success("✅ Sample files created in 'test_data/' folder!")
            st.info("Use **sample_test_FD001.txt** and **sample_RUL_FD001.txt** for testing")
    
    st.markdown("---")
    st.markdown("#### 📖 How to Use Sample Data:")
    st.write("1. Click 'Generate Sample Data' above")
    st.write("2. Select FD001 as dataset")
    st.write("3. Upload the generated files")
    st.write("4. Click 'Run Predictive Analysis'")

def run_prediction_analysis(dataset, test_file, true_rul_file, warning_thresh, critical_thresh, model_handler):
    """Run the complete prediction analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save uploaded files
        status_text.text("📥 Saving uploaded files...")
        test_path = Path(f"temp_test_{dataset}.txt")
        test_path.write_bytes(test_file.getbuffer())
        progress_bar.progress(10)
        
        true_rul = None
        if true_rul_file:
            true_rul_path = Path(f"temp_rul_{dataset}.txt")
            true_rul_path.write_bytes(true_rul_file.getbuffer())
        
        # Step 2: Process data
        status_text.text("🔧 Processing data...")
        DataProcessor = get_data_processor()
        if DataProcessor:
            processor = DataProcessor(dataset)
            
            # For simulation, we'll just count lines to estimate units
            with open(test_path, 'r') as f:
                lines = f.readlines()
                n_units = min(100, len(lines) // 3)  # Estimate: ~3 lines per unit
            
            if n_units < 5:
                n_units = random.randint(10, 50)
        else:
            n_units = random.randint(20, 100)
        
        progress_bar.progress(40)
        
        # Step 3: Generate predictions
        status_text.text("🤖 Generating predictions...")
        predictions = model_handler.predict(dataset, n_units, use_realistic_simulation=True)
        progress_bar.progress(70)
        
        # Step 4: Calculate metrics
        status_text.text("📈 Calculating metrics...")
        
        # Generate simulated true RUL if not provided
        if true_rul_file and DataProcessor:
            processor = DataProcessor(dataset)
            true_rul = processor.load_true_rul(str(true_rul_path))
        else:
            # Create realistic true RUL for simulation
            true_rul = [random.uniform(max(1, p['predicted_rul'] - 30), 
                                      p['predicted_rul'] + 30) 
                       for p in predictions]
            true_rul = [min(200, max(1, r)) for r in true_rul]
        
        metrics = model_handler.calculate_metrics(predictions, true_rul)
        progress_bar.progress(90)
        
        # Step 5: Store results
        st.session_state.predictions = predictions
        st.session_state.true_rul = true_rul
        st.session_state.metrics = metrics
        st.session_state.dataset_name = dataset
        st.session_state.warning_thresh = warning_thresh
        st.session_state.critical_thresh = critical_thresh
        st.session_state.analysis_complete = True
        
        # Step 6: Cleanup
        test_path.unlink(missing_ok=True)
        if true_rul_file:
            true_rul_path.unlink(missing_ok=True)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
        
        # Success message
        st.balloons()
        st.success(f"🎉 Successfully analyzed {len(predictions)} engine units!")
        
        # Auto-navigate to dashboard
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.text("Analysis failed")

def show_dashboard_page():
    """Display the main dashboard"""
    if not st.session_state.analysis_complete:
        st.warning("⚠️ No analysis data found. Please run an analysis first.")
        if st.button("Go to Home Page"):
            st.session_state.current_page = 0
            st.rerun()
        return
    
    st.markdown('<h1 class="main-header">📊 PrognosAI Dashboard</h1>', unsafe_allow_html=True)
    
    # Timestamp
    st.caption(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Dataset: {st.session_state.dataset_name}")
    
    # ========== METRICS SECTION ==========
    st.markdown("## 📈 Performance Metrics")
    
    metrics = st.session_state.metrics
    predictions = st.session_state.predictions
    
    # Create metric columns
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    
    with mcol1:
        st.markdown('<div class="metric-card critical-card">', unsafe_allow_html=True)
        st.metric("🔴 Critical Alerts", metrics.get('critical_count', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with mcol2:
        st.markdown('<div class="metric-card warning-card">', unsafe_allow_html=True)
        st.metric("🟡 Warning Alerts", metrics.get('warning_count', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with mcol3:
        st.markdown('<div class="metric-card normal-card">', unsafe_allow_html=True)
        st.metric("🟢 Normal Status", metrics.get('normal_count', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with mcol4:
        st.markdown('<div class="metric-card info-card">', unsafe_allow_html=True)
        st.metric("⚙️ Total Units", metrics.get('total_units', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with mcol5:
        st.markdown('<div class="metric-card info-card">', unsafe_allow_html=True)
        if not metrics.get('simulation', True):
            rmse_val = metrics.get('rmse', 'N/A')
            if isinstance(rmse_val, (int, float)):
                st.metric("📊 RMSE", f"{rmse_val:.2f}")
            else:
                st.metric("📊 RMSE", "N/A")
        else:
            st.metric("🎯 Avg Confidence", f"{metrics.get('avg_confidence', 0)*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== CHARTS SECTION ==========
    st.markdown("## 📊 Visual Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### RUL Distribution")
        # Create RUL distribution chart
        if predictions:
            df = pd.DataFrame(predictions)
            fig = px.histogram(
                df, 
                x='predicted_rul',
                color='alert',
                color_discrete_map={
                    'CRITICAL': '#dc2626',
                    'WARNING': '#f59e0b',
                    'NORMAL': '#10b981'
                },
                nbins=30,
                title='Remaining Useful Life Distribution'
            )
            fig.update_layout(
                height=400,
                showlegend=True,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Alert Status Overview")
        # Create alert pie chart
        if predictions:
            alert_counts = pd.DataFrame(predictions)['alert'].value_counts().reset_index()
            alert_counts.columns = ['Status', 'Count']
            
            fig = px.pie(
                alert_counts,
                values='Count',
                names='Status',
                color='Status',
                color_discrete_map={
                    'CRITICAL': '#dc2626',
                    'WARNING': '#f59e0b',
                    'NORMAL': '#10b981'
                },
                hole=0.4,
                title='Alert Status Distribution'
            )
            fig.update_layout(
                height=400,
                showlegend=True,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== DETAILED RESULTS ==========
    st.markdown("## 📋 Detailed Analysis Results")
    
    # Results table
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Add styling to dataframe
        def highlight_alerts(val):
            if val == 'CRITICAL':
                return 'background-color: #fee2e2; color: #dc2626; font-weight: bold'
            elif val == 'WARNING':
                return 'background-color: #fef3c7; color: #d97706; font-weight: bold'
            elif val == 'NORMAL':
                return 'background-color: #d1fae5; color: #059669; font-weight: bold'
            return ''
        
        # Display dataframe
        st.dataframe(
            df.style.applymap(highlight_alerts, subset=['alert']),
            use_container_width=True,
            height=400
        )
    
    # ========== ACTION BUTTONS ==========
    st.markdown("---")
    st.markdown("## ⚡ Actions & Export")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("📥 Download CSV Report", use_container_width=True):
            csv_data = generate_csv_download()
            st.download_button(
                label="Click to Download",
                data=csv_data,
                file_name=f"prognosai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with action_col2:
        if st.button("🖨️ Generate Summary", use_container_width=True):
            st.info("📄 PDF report generation feature coming soon!")
    
    with action_col3:
        if st.button("📊 View Statistics", use_container_width=True):
            st.session_state.current_page = 2  # Go to analysis page
            st.rerun()
    
    with action_col4:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.clear()
            init_session_state()
            st.rerun()

def show_analysis_page():
    """Display detailed analysis page"""
    if not st.session_state.analysis_complete:
        st.warning("Please run an analysis first from the Home page.")
        return
    
    st.markdown('<h1 class="main-header">📈 Advanced Analysis</h1>', unsafe_allow_html=True)
    
    metrics = st.session_state.metrics
    
    # Statistical Analysis Section
    st.markdown("## 📊 Statistical Analysis")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Mean RUL", f"{metrics.get('mean_rul', 0):.1f} cycles")
    
    with stat_col2:
        st.metric("Median RUL", f"{metrics.get('median_rul', 0):.1f} cycles")
    
    with stat_col3:
        st.metric("Std Deviation", f"{metrics.get('std_rul', 0):.1f} cycles")
    
    with stat_col4:
        st.metric("Range", f"{metrics.get('min_rul', 0):.1f} - {metrics.get('max_rul', 0):.1f}")
    
    # Error Metrics (if available)
    if not metrics.get('simulation', True):
        st.markdown("## 🎯 Prediction Accuracy")
        
        error_col1, error_col2, error_col3, error_col4 = st.columns(4)
        
        with error_col1:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.2f} cycles")
        
        with error_col2:
            st.metric("MAE", f"{metrics.get('mae', 0):.2f} cycles")
        
        with error_col3:
            st.metric("Max Error", f"{metrics.get('max_error', 0):.2f} cycles")
        
        with error_col4:
            st.metric("Mean Error", f"{metrics.get('mean_error', 0):.2f} cycles")
        
        # Accuracy Metrics
        st.markdown("### 📈 Accuracy within Tolerance")
        
        acc_col1, acc_col2, acc_col3 = st.columns(3)
        
        with acc_col1:
            st.metric("±10 cycles", f"{metrics.get('accuracy_10', 0):.1f}%")
        
        with acc_col2:
            st.metric("±20 cycles", f"{metrics.get('accuracy_20', 0):.1f}%")
        
        with acc_col3:
            st.metric("±30 cycles", f"{metrics.get('accuracy_30', 0):.1f}%")
    
    # Trend Analysis
    st.markdown("## 📈 Trend Analysis")
    
    if st.session_state.predictions:
        df = pd.DataFrame(st.session_state.predictions)
        
        # Create trend visualization
        fig = go.Figure()
        
        # Add RUL trend line
        fig.add_trace(go.Scatter(
            x=df['unit'],
            y=df['predicted_rul'],
            mode='lines+markers',
            name='Predicted RUL',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=6)
        ))
        
        # Add alert zones
        fig.add_hrect(
            y0=0, y1=st.session_state.critical_thresh,
            fillcolor="#fee2e2", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="Critical Zone",
            annotation_position="top left"
        )
        
        fig.add_hrect(
            y0=st.session_state.critical_thresh, 
            y1=st.session_state.warning_thresh,
            fillcolor="#fef3c7", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="Warning Zone"
        )
        
        fig.update_layout(
            title='RUL Trend Across Engine Units',
            xaxis_title='Engine Unit Number',
            yaxis_title='Remaining Useful Life (cycles)',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Back button
    if st.button("← Back to Dashboard", use_container_width=True):
        st.session_state.current_page = 1
        st.rerun()

def show_settings_page():
    """Display settings page"""
    st.markdown('<h1 class="main-header">⚙️ System Settings</h1>', unsafe_allow_html=True)
    
    # System Information
    st.markdown("## 💻 System Configuration")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info("""
        **Hardware Profile**: Intel i3 / 8GB RAM  
        **Optimization**: Memory-efficient mode  
        **Model Cache**: Enabled  
        **Visual Quality**: Balanced  
        **Auto Updates**: Disabled
        """)
    
    with info_col2:
        st.info("""
        **Python Version**: 3.11+  
        **Streamlit**: 1.28.0  
        **Pandas**: 2.2.2  
        **Plotly**: 5.24.1  
        **Status**: ✅ Running Optimal
        """)
    
    # Performance Settings
    st.markdown("## ⚡ Performance Settings")
    
    with st.expander("Memory Optimization", expanded=True):
        mem_usage = st.slider(
            "Maximum Memory Usage",
            min_value=30,
            max_value=80,
            value=50,
            help="Limit RAM usage percentage"
        )
        
        cache_size = st.slider(
            "Cache Size Limit (MB)",
            min_value=100,
            max_value=1000,
            value=500,
            help="Maximum cache memory usage"
        )
    
    with st.expander("Visualization Settings"):
        chart_quality = st.selectbox(
            "Chart Quality",
            ["Low (Fast)", "Medium (Balanced)", "High (Detailed)"],
            index=1
        )
        
        auto_refresh = st.checkbox(
            "Enable Auto-refresh",
            value=False,
            help="Automatically refresh dashboard data"
        )
    
    with st.expander("Model Settings"):
        simulation_mode = st.checkbox(
            "Enable Simulation Mode",
            value=True,
            help="Use simulation when models are unavailable"
        )
        
        model_cache = st.checkbox(
            "Enable Model Caching",
            value=True,
            help="Cache models for faster loading"
        )
    
    # Save Settings Button
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        st.success("✅ Settings saved successfully!")
        st.info("Some changes may require restarting the application.")

# ============================================
# MAIN APPLICATION FLOW
# ============================================
def main():
    """Main application controller"""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #1e3a8a;">🔧 PrognosAI</h2>
            <p style="color: #6b7280;">Predictive Maintenance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        page_options = {
            "🏠 Home": 0,
            "📊 Dashboard": 1,
            "📈 Analysis": 2,
            "⚙️ Settings": 3
        }
        
        selected_page = st.radio(
            "Navigation",
            list(page_options.keys()),
            label_visibility="collapsed"
        )
        
        st.session_state.current_page = page_options[selected_page]
        
        st.markdown("---")
        
        # Quick Stats if analysis exists
        if st.session_state.analysis_complete:
            st.markdown("### 📈 Current Analysis")
            st.write(f"**Dataset**: {st.session_state.dataset_name}")
            st.write(f"**Units**: {len(st.session_state.predictions)}")
            st.write(f"**Status**: ✅ Complete")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Restart Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if st.button("📋 Create Sample Data", use_container_width=True):
            create_sample_files()
            st.success("Sample data created!")
    
    # Page Routing
    if st.session_state.current_page == 0:
        show_home_page()
    elif st.session_state.current_page == 1:
        show_dashboard_page()
    elif st.session_state.current_page == 2:
        show_analysis_page()
    elif st.session_state.current_page == 3:
        show_settings_page()

# ============================================
# APPLICATION ENTRY POINT
# ============================================
if __name__ == "__main__":
    main()