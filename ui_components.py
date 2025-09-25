import streamlit as st
from pathlib import Path

def load_css():
    """Load the main CSS file into Streamlit."""
    css_file = Path("assets/styles.css")
    
    if css_file.exists():
        with open(css_file) as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Using default Streamlit styling.")

def load_template(template_name: str) -> str:
    """Load an HTML template file."""
    template_file = Path(f"templates/{template_name}.html")
    
    if template_file.exists():
        with open(template_file, 'r') as f:
            return f.read()
    else:
        st.error(f"Template {template_name}.html not found")
        return ""

def render_header():
    """Render the application header."""
    header_html = load_template("header")
    st.markdown(header_html, unsafe_allow_html=True)

def render_footer():
    """Render the application footer."""
    footer_html = load_template("footer")
    st.markdown("---")
    st.markdown(footer_html, unsafe_allow_html=True)

def render_sentiment_card(sentiment: str, emoji: str):
    """Render a sentiment analysis result card."""
    sentiment_class = f"{sentiment.lower()}-sentiment"
    
    template = load_template("sentiment_card")
    formatted_html = template.format(
        sentiment_class=sentiment_class,
        emoji=emoji,
        sentiment_label=sentiment.upper()
    )
    
    st.markdown(formatted_html, unsafe_allow_html=True)

def render_similarity_card(rank: int, similarity_score: float, text: str):
    """Render a similarity search result card."""
    # Determine score color
    if similarity_score > 0.7:
        score_color = "#28a745"
    elif similarity_score > 0.5:
        score_color = "#ffc107"
    else:
        score_color = "#dc3545"
    
    template = load_template("similarity_card")
    formatted_html = template.format(
        rank=rank,
        score_color=score_color,
        similarity_score=f"{similarity_score:.4f}",
        text_content=text
    )
    
    st.markdown(formatted_html, unsafe_allow_html=True)

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment."""
    emojis = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emojis.get(sentiment.lower(), '❓')

def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment display."""
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545', 
        'neutral': '#ffc107'
    }
    return colors.get(sentiment.lower(), '#6c757d')

def create_info_box(message: str, box_type: str = "info"):
    """Create an info/warning/error box."""
    box_class = f"{box_type}-box"
    
    html = f"""
    <div class="{box_class}">
        <p style="margin: 0;">{message}</p>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)