import streamlit as st
from datetime import datetime
import os

def load_css(css_file):
    """Load CSS file for styling"""
    try:
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Create basic styling if CSS file doesn't exist
        st.markdown("""
        <style>
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #eee;
        }
        .welcome-container {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin: 2rem 0;
        }
        .features {
            text-align: left;
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, datetime):
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return str(timestamp)

def setup_directories():
    """Setup required directories"""
    directories = ['database', 'static', 'components', 'utils']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
