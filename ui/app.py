# ui/app.py
"""
Fast Streamlit UI for Sentra Log Analysis
- Optimized imports and caching
- Real-time analytics dashboard  
- Fast file processing with progress
- Clean, responsive interface
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import backend
except ImportError as e:
    st.error(f"❌ Failed to import backend: {e}")
    st.error(f"Make sure you're running from the project root directory")
    st.stop()

# Page config
st.set_page_config(
    page_title="🔍 Sentra - Log Investigator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.sub-header {
    color: #666;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.result-item {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f9f9f9;
}
.status-ok { border-left: 4px solid #28a745; }
.status-warning { border-left: 4px solid #ffc107; }
.status-error { border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_stats():
    """Get cached system statistics"""
    return backend.get_stats()

@st.cache_data(ttl=300)
def get_cached_dashboard():
    """Get cached dashboard data"""
    return backend.get_dashboard_data()

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="main-header">🔍 Sentra Log Investigator</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Fast semantic search and analysis for log files</div>', unsafe_allow_html=True)
    
    with col2:
        stats = get_cached_stats()
        if stats.get('has_index'):
            st.success(f"✓ {stats['total_entries']:,} entries indexed")
        else:
            st.info("Upload logs to get started")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Controls")
        
        # File upload
        st.markdown("### 📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose log files",
            accept_multiple_files=True,
            type=['log', 'txt', 'csv'],
            help="Upload .log, .txt, or .csv files"
        )
        
        if uploaded_files:
            process_btn = st.button("🚀 Process Files", type="primary", use_container_width=True)
            if process_btn:
                process_uploaded_files(uploaded_files)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        show_raw = st.checkbox("Show raw log lines", value=False)
        max_results = st.slider("Max search results", 5, 50, 10)
        
        # Quick stats in sidebar
        if stats.get('has_index'):
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Entries", f"{stats['total_entries']:,}")
            st.metric("Log Files", stats['total_files'])
            st.metric("Unique IPs", stats['unique_ips'])
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Dashboard", "📈 Analytics"])
    
    with tab1:
        search_interface(max_results, show_raw)
    
    with tab2:
        dashboard_interface()
    
    with tab3:
        analytics_interface()

def process_uploaded_files(uploaded_files):
    """Process uploaded files with progress tracking"""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save files
    saved_paths = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        file_path = upload_dir / file.name
        with open(file_path, 'wb') as f:
            f.write(file.read())
        saved_paths.append(str(file_path))
        
        progress = (i + 1) / len(uploaded_files) * 0.5  # First 50% for saving
        progress_bar.progress(progress)
        status_text.text(f"Saving {file.name}...")
    
    # Process files
    status_text.text("Processing and indexing...")
    try:
        total_entries, indexed = backend.process_files(saved_paths)
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        if total_entries > 0:
            st.success(f"✅ Processed {total_entries:,} entries from {len(uploaded_files)} files")
            # Clear cache to refresh stats
            get_cached_stats.clear()
            get_cached_dashboard.clear()
            st.rerun()
        else:
            st.warning("⚠️ No entries found in uploaded files")
            
    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        progress_bar.empty()
        status_text.empty()

def search_interface(max_results, show_raw):
    """Search interface"""
    st.markdown("### 🔍 Semantic Search")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        default_query = backend.get_last_query() or ""
        query = st.text_input(
            "Search query",
            value=default_query,
            placeholder="e.g., 'failed login attempts', 'high response time', 'error 500'",
            help="Use natural language to describe what you're looking for"
        )
    
    with col2:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Quick search buttons
    st.markdown("**Quick searches:**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    with quick_col1:
        if st.button("🚨 Errors", use_container_width=True):
            query = "error failed exception"
    with quick_col2:
        if st.button("🔒 Security", use_container_width=True):
            query = "login authentication unauthorized"
    with quick_col3:
        if st.button("⚡ Performance", use_container_width=True):
            query = "slow timeout high response time"
    with quick_col4:
        if st.button("📊 Status Codes", use_container_width=True):
            query = "404 500 503 error status"
    
    # Perform search
    if (search_btn or query != default_query) and query.strip():
        backend.remember_query(query)
        
        with st.spinner("Searching..."):
            results = backend.search_logs(query, max_results)
        
        if results:
            # Analysis summary
            analysis = backend.analyze_results(results)
            st.info(f"📋 **Analysis:** {analysis}")
            
            # Results
            st.markdown(f"### 📄 Results ({len(results)})")
            
            for i, result in enumerate(results):
                # Determine status color
                status = result.get('status', '000')
                if status.startswith('2'):
                    status_class = "status-ok"
                elif status.startswith('4') or status.startswith('5'):
                    status_class = "status-error"
                else:
                    status_class = "status-warning"
                
                with st.container():
                    st.markdown(f'<div class="result-item {status_class}">', unsafe_allow_html=True)
                    
                    # Header row
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        method = result.get('method', 'N/A')
                        endpoint = result.get('endpoint', 'N/A')
                        st.markdown(f"**{method}** `{endpoint}`")
                    
                    with col2:
                        if result.get('ip'):
                            st.markdown(f"🌐 **IP:** {result['ip']}")
                    
                    with col3:
                        if result.get('status'):
                            st.markdown(f"📊 **Status:** {result['status']}")
                    
                    # Details row
                    details_col1, details_col2 = st.columns([1, 1])
                    with details_col1:
                        if result.get('timestamp'):
                            st.caption(f"⏰ {result['timestamp']}")
                    with details_col2:
                        st.caption(f"📁 {result.get('source_file', 'unknown')}")
                    
                    # Show raw log if requested
                    if show_raw and st.button(f"Show raw log", key=f"raw_{i}"):
                        st.code(f"Line {result.get('line_number', '?')}: [raw log content not stored]")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("🤷 No results found. Try different search terms.")

def dashboard_interface():
    """Dashboard with analytics"""
    st.markdown("### 📊 Dashboard")
    
    dashboard_data = get_cached_dashboard()
    
    if 'error' in dashboard_data:
        st.info("📁 Upload and process log files to see analytics")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entries", f"{dashboard_data['total_entries']:,}")
    with col2:
        st.metric("Unique IPs", len(dashboard_data.get('top_ips', {})))
    with col3:
        st.metric("Log Files", len(dashboard_data.get('file_stats', {})))
    with col4:
        st.metric("Status Codes", len(dashboard_data.get('status_codes', {})))
    
    # Charts row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 🌐 Top IP Addresses")
        if dashboard_data.get('top_ips'):
            ip_df = pd.DataFrame(list(dashboard_data['top_ips'].items()), 
                               columns=['IP', 'Count'])
            fig = px.bar(ip_df.head(10), x='Count', y='IP', orientation='h',
                        color='Count', color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No IP data available")
    
    with chart_col2:
        st.markdown("#### 📊 HTTP Status Codes")
        if dashboard_data.get('status_codes'):
            status_df = pd.DataFrame(list(dashboard_data['status_codes'].items()),
                                   columns=['Status', 'Count'])
            fig = px.pie(status_df, values='Count', names='Status',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status code data available")
    
    # File statistics table
    if dashboard_data.get('file_stats'):
        st.markdown("#### 📁 File Statistics")
        file_df = pd.DataFrame(list(dashboard_data['file_stats'].items()),
                              columns=['File', 'Entries'])
        file_df = file_df.sort_values('Entries', ascending=False)
        st.dataframe(file_df, use_container_width=True)

def analytics_interface():
    """Advanced analytics interface"""
    st.markdown("### 📈 Advanced Analytics")
    
    dashboard_data = get_cached_dashboard()
    
    if 'error' in dashboard_data:
        st.info("📁 Upload and process log files to see advanced analytics")
        return
    
    # Time series placeholder (would need timestamp parsing improvements)
    st.markdown("#### ⏰ Timeline Analysis")
    st.info("Timeline analysis will be available once timestamp parsing is enhanced")
    
    # Security insights
    st.markdown("#### 🔒 Security Insights")
    
    if dashboard_data.get('top_ips'):
        # Simple anomaly detection based on request counts
        ip_counts = list(dashboard_data['top_ips'].values())
        if len(ip_counts) > 1:
            mean_requests = sum(ip_counts) / len(ip_counts)
            threshold = mean_requests * 3  # Simple 3x threshold
            
            suspicious_ips = [(ip, count) for ip, count in dashboard_data['top_ips'].items() 
                            if count > threshold]
            
            if suspicious_ips:
                st.warning(f"🚨 **{len(suspicious_ips)} potentially suspicious IPs detected** (>3x average requests)")
                for ip, count in suspicious_ips[:5]:
                    st.write(f"- {ip}: {count:,} requests")
            else:
                st.success("✅ No obviously suspicious IP activity detected")
    
    # Error analysis
    if dashboard_data.get('status_codes'):
        error_codes = {k: v for k, v in dashboard_data['status_codes'].items() 
                      if k.startswith('4') or k.startswith('5')}
        
        if error_codes:
            st.markdown("#### ❌ Error Code Analysis")
            total_errors = sum(error_codes.values())
            total_requests = sum(dashboard_data['status_codes'].values())
            error_rate = (total_errors / total_requests) * 100 if total_requests > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Errors", f"{total_errors:,}")
            with col2:
                st.metric("Error Rate", f"{error_rate:.1f}%")
            
            # Error breakdown
            error_df = pd.DataFrame(list(error_codes.items()), 
                                  columns=['Status Code', 'Count'])
            fig = px.bar(error_df, x='Status Code', y='Count', 
                        title="Error Codes Distribution",
                        color='Count', color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()