import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Set page config for a professional look
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")

# CSS for styling with new design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Inter:wght@400;500&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
    
    body, .main {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%) !important; /* New Background Light to Darker/Subtle */
        color: #212529 !important; /* New Text Primary */
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        color: #0A74DA; /* New Primary Blue for headings */
    }
    
    
    /* Home page styling */
    .home-background {
        background: linear-gradient(135deg, #0A74DA 0%, #055cb5 100%); /* Primary Blue gradient */
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        z-index: -1;
        opacity: 0.1;
    }
    .main-content {
        text-align: center;
        color: #212529; /* New Text Primary */
        padding: 3rem;
    }
    .main-content h1 {
        font-size: 3.5rem;
        font-weight: 700;
        color: #0A74DA; /* New Primary Blue */
        font-family: 'Montserrat', sans-serif; /* New Headings Font */
    }
    .main-content .slogan {
        font-size: 1.8rem;
        font-weight: 500;
        margin: 1rem 0;
        color: #495057; /* New Text Secondary */
        font-family: 'Inter', sans-serif; /* New Body Font */
    }
    /* Button styling */
    .stButton>button {
        background-color: #0A74DA; /* New Primary Blue */
        color: #FFFFFF; /* White text */
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker shade of Primary Blue */
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stButton>button:active {
        background-color: #004085; /* Even darker shade for active state */
        transform: translateY(0px);
    }
    
    /* Input field styling */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #212529 !important; /* New Text Primary */
        border-radius: 8px !important;
        border: 1px solid #E9ECEF !important; /* Subtle border */
        padding: 10px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox div[data-baseweb="popover"] ul {
        background-color: #FFFFFF !important;
        color: #212529 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg { /* This is a Streamlit specific class, might need to be updated if Streamlit changes */
        background-color: #E9ECEF !important; /* New Background Darker/Subtle */
        border-right: 1px solid #D1D5DB;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content .stMarkdown {
        font-family: 'Montserrat', sans-serif;
    }
    .sidebar .sidebar-content .stMarkdown p, .sidebar .sidebar-content .stRadio label span {
        color: #212529 !important; /* New Text Primary for sidebar text */
        font-family: 'Inter', sans-serif;
    }
    
    /* Success and Alert messages */
    .stSuccess {
        background-color: #28A745 !important; /* New Success Green */
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif;
    }
    .stAlert {
        background-color: #DC3545 !important; /* New Error Red */
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Icon text styling */
    .icon-text {
        display: flex;
        align-items: center;
        gap: 10px; /* Increased gap */
        font-size: 16px;
        margin-bottom: 10px; /* Increased margin */
        color: #495057; /* New Text Secondary */
        font-family: 'Inter', sans-serif;
    }
    .icon-text i {
        font-size: 1.2em; /* Slightly larger icons */
    }
    .icon-text .fas.fa-check-circle {
        color: #28A745; /* New Success Green */
    }
    .icon-text .fas.fa-chart-line, .icon-text .fas.fa-wallet, .icon-text .fas.fa-search, .icon-text .fas.fa-keyboard, .icon-text .fas.fa-money-bill-wave {
        color: #FF7F00; /* New Accent Orange */
    }
    .icon-text .fas.fa-info-circle {
        color: #0A74DA; /* New Primary Blue for info */
    }
    .icon-text .fas.fa-plane, .icon-text .fas.fa-ticket-alt, .icon-text .fas.fa-rupee-sign, .icon-text .fas.fa-route, .icon-text .fas.fa-star, .icon-text .fas.fa-exclamation-circle, .icon-text .fas.fa-city, .icon-text .fas.fa-chair, .icon-text .fas.fa-plane-arrival, .icon-text .fas.fa-calendar {
        color: #495057; /* New Text Secondary for general icons */
    }
    
    /* Specific Icon Colors for Titles / Headers */
    .sidebar .icon-text i.fa-compass, .sidebar .icon-text i.fa-plane {
        color: #0A74DA; /* Primary Blue for sidebar navigation icon */
    }
    .main-content h1 i.fa-plane-departure {
        color: #0A74DA; /* Primary Blue for main title icon */
    }
    h2 i.fa-chart-pie, h2 i.fa-chart-bar, h2 i.fa-analytics, h2 i.fa-map-marked-alt {
        color: #0A74DA; /* Primary Blue for analytics header icon */
    }
    h2 i.fa-calculator, h2 i.fa-magic, h2 i.fa-search-dollar {
        color: #0A74DA; /* Primary Blue for predict price header icon */
    }
    h2 i.fa-lightbulb, h2 i.fa-map-signs, h2 i.fa-suitcase-rolling {
        color: #0A74DA; /* Primary Blue for travel corner header icon */
    }

    /* Updated Names Container Styling */
    .names-container-wrapper { /* Wrapper to center the names-container */
        display: flex;
        justify-content: center;
        margin-top: 3rem;
        margin-bottom: 2rem;
    }
    .names-container {
        padding: 2rem;
        background: #FFFFFF;
        border-radius: 16px;
        display: flex; 
        flex-direction: column;
        gap: 1.5rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
        text-align: left;
        max-width: 800px; 
        width: 100%; /* Ensure it takes up available width up to max-width */
    }

    .team-section h3 {
        font-size: 1.8rem;
        color: #0A74DA;
        font-family: 'Montserrat', sans-serif;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .team-section h3 i {
        font-size: 1.6rem;
    }

    .name-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
    }

    .name-card {
        background-color: #F8F9FA;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 500;
        color: #212529;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.07);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .name-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 5px 12px rgba(0,0,0,0.12);
    }

    .name-card i {
        color: #FF7F00; /* Accent Orange */
        font-size: 1.3rem;
    }
    /* End of Updated Names Container Styling */

    /* Responsive design */
    @media (max-width: 768px) {
        .main-content h1 { font-size: 2.8rem; }
        .main-content .slogan { font-size: 1.6rem; }
        .icon-text { font-size: 15px; }
        .names-container-wrapper {
             margin-top: 2rem;
        }
        .names-container {
            padding: 1.5rem; /* Adjust padding for smaller screens */
        }
        .team-section h3 {
            font-size: 1.5rem; /* Adjust heading size */
        }
        .name-card {
            font-size: 1rem; /* Adjust card font size */
            padding: 1rem;
        }
        .name-grid {
            grid-template-columns: 1fr; /* Stack cards on smaller screens */
        }
    }
    
    /* Insights box styling */
    .insights-box {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 25px;
        border: 1px solid #E9ECEF;
    }
    
    /* Selections box styling */
    .selections-box {
        background-color: #FFFFFF;
        color: #212529;
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-top: 25px;
    }

    /* Make Streamlit headers use Montserrat and Primary Blue */
    div[data-testid="stHeading"] h1, div[data-testid="stHeading"] h2, div[data-testid="stHeading"] h3, div[data-testid="stHeading"] h4, div[data-testid="stHeading"] h5, div[data-testid="stHeading"] h6 {
        font-family: 'Montserrat', sans-serif !important;
        color: #0A74DA !important; /* Primary Blue */
    }
    /* Adjust Streamlit subheader styling */
    div[data-testid="stSubheader"] {
        font-family: 'Montserrat', sans-serif !important;
        color: #495057 !important; /* Text Secondary */
        font-size: 1.25rem; /* Example size, adjust as needed */
        font-weight: 600;
    }

    /* Aggressive global overflow fix attempt */
    div, section, article, main, .main, .block-container, [data-testid*="Block"], [data-testid*="Selectbox"], [data-testid*="Input"] {
        overflow: visible !important;
    }

    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_flight_data():
    try:
        df = pd.read_csv('cleaned_flight_data.csv')
        required_columns = ['airline', 'From', 'to', 'departure_time', 'arrival_time', 
                        'stops', 'Class', 'price', 'days_left', 'time_taken']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns in flight data: {missing_columns}")
            st.stop()
        return df
    except FileNotFoundError:
        st.error("Flight data file (cleaned_flight_data.csv) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading flight data: {str(e)}")
        st.stop()
# Process data
@st.cache_data
def process_flight_data(df):
    try:
        le = LabelEncoder()
        df['airline_encoded'] = le.fit_transform(df['airline'])
        df['From_encoded'] = le.fit_transform(df['From'])
        df['to_encoded'] = le.fit_transform(df['to'])
        df['departure_time_encoded'] = le.fit_transform(df['departure_time'])
        df['arrival_time_encoded'] = le.fit_transform(df['arrival_time'])
        stops_mapping = {'Zero': 0, 'One': 1, 'Two or more': 2}
        df['stops_encoded'] = df['stops'].map(stops_mapping)
        class_mapping = {'Economy': 0, 'Business': 1}
        df['Class_encoded'] = df['Class'].map(class_mapping)
        return df
    except Exception as e:
        st.error(f"Error processing flight data: {str(e)}")
        st.stop()

# Load model
@st.cache_resource
def load_model():
    try:
        import os
        file_path = 'FPP_model.pkl'
        if not os.path.exists(file_path):
            st.error(f"Model file {file_path} not found in path: {os.getcwd()}")
            st.stop()
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Main data loading
df_flights = load_flight_data()
if df_flights is not None and not df_flights.empty:
    processed_df_flights = process_flight_data(df_flights.copy()) # Use a copy for processing
else:
    processed_df_flights = pd.DataFrame() # Ensure it's a DataFrame even if loading fails

model = load_model()

# Sidebar
st.sidebar.title("App Navigation") 
st.sidebar.markdown('<div class="icon-text"><i class="fas fa-compass"></i> Flight Predictor Suite</div>', unsafe_allow_html=True)

page_options = ["Home", "Analytics for Business", "Predict Price for Business", "Traveler Corner"]
page = st.sidebar.radio("Go to", page_options)

# Home page
if page == "Home":
    st.markdown("""
    <div class="home-background"></div>
    <div class="main-content">
        <h1><i class="fas fa-plane-departure"></i> Flight Price Prediction</h1>
        <p class="slogan">Plan smarter, travel cheaper, fly happier.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="names-container-wrapper">
        <div class="names-container">
            <div class="team-section">
                <h3><i class="fas fa-users"></i>Developed By</h3>
                <div class="name-grid">
                    <div class="name-card"><i class="fas fa-user-tie"></i> Omar Fayad</div>
                    <div class="name-card"><i class="fas fa-user-tie"></i> Ahmed Magdy</div>
                    <div class="name-card"><i class="fas fa-user-tie"></i> Mahmoud Hamdy</div>
                </div>
            </div>
            <div class="team-section">
                <h3><i class="fas fa-chalkboard-teacher"></i>Supervised By</h3>
                <div class="name-grid" style="grid-template-columns: 1fr;">
                    <div class="name-card"><i class="fas fa-user-graduate"></i> Dr. Hewayda Mohamed</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Why Use This?") 
        st.markdown('<div class="icon-text"><i class="fas fa-check-circle"></i> Accurate: AI-powered price predictions.</div>', unsafe_allow_html=True)
        st.markdown('<div class="icon-text"><i class="fas fa-chart-line"></i> Insightful: Deep analytics for smarter decisions.</div>', unsafe_allow_html=True)
        st.markdown('<div class="icon-text"><i class="fas fa-wallet"></i> Savings: Tailored tips to cut costs.</div>', unsafe_allow_html=True)
    with col2:
        st.subheader("How to Start") 
        st.markdown('<div class="icon-text"><i class="fas fa-search"></i> Explore trends in "Analytics for Business".</div>', unsafe_allow_html=True)
        st.markdown('<div class="icon-text"><i class="fas fa-keyboard"></i> Enter details in "Predict Price for Business".</div>', unsafe_allow_html=True)
        st.markdown('<div class="icon-text"><i class="fas fa-money-bill-wave"></i> Save big with our insights.</div>', unsafe_allow_html=True)

# Analytics page
elif page == "Analytics for Business":
    st.header("Flight Data Analytics")
    st.markdown('<div class="icon-text" style="color: #0A74DA;"><i class="fas fa-chart-pie"></i> Discover trends to book the best flights at the best prices.</div>', unsafe_allow_html=True)
    
    if df_flights is None or df_flights.empty:
        st.error("Flight data is not available. Cannot display analytics.")
    else:
        st.subheader("General Insights")
        st.markdown('<div class="insights-box">', unsafe_allow_html=True)
        num_airlines = df_flights['airline'].nunique()
        num_flights = len(df_flights)
        avg_price = df_flights['price'].mean()
        
        most_popular_route_str = "N/A"
        if not df_flights.empty:
            most_popular_route_group = df_flights.groupby(['From', 'to']).size()
            if not most_popular_route_group.empty:
                most_popular_route = most_popular_route_group.idxmax()
                most_popular_route_str = f"{most_popular_route[0]} to {most_popular_route[1]}"
        
        cheapest_airline_str = "N/A"
        if not df_flights.empty:
            cheapest_airline_group = df_flights.groupby('airline')['price'].mean()
            if not cheapest_airline_group.empty:
                cheapest_airline = cheapest_airline_group.idxmin()
                cheapest_airline_price = cheapest_airline_group.min()
                cheapest_airline_str = f"{cheapest_airline} (₹{cheapest_airline_price:,.2f})"

        most_expensive_airline_str = "N/A"
        if not df_flights.empty:
            most_expensive_airline_group = df_flights.groupby('airline')['price'].mean()
            if not most_expensive_airline_group.empty:
                most_expensive_airline = most_expensive_airline_group.idxmax()
                most_expensive_airline_price = most_expensive_airline_group.max()
                most_expensive_airline_str = f"{most_expensive_airline} (₹{most_expensive_airline_price:,.2f})"
            
        num_cities = df_flights['From'].nunique()
        st.markdown(f"""
        - <div class="icon-text"><i class="fas fa-plane"></i> Number of Airlines: {num_airlines}</div>
        - <div class="icon-text"><i class="fas fa-ticket-alt"></i> Total Number of Flights: {num_flights}</div>
        - <div class="icon-text"><i class="fas fa-rupee-sign"></i> Average Flight Price: ₹{avg_price:,.2f}</div>
        - <div class="icon-text"><i class="fas fa-route"></i> Most Popular Route: {most_popular_route_str}</div>
        - <div class="icon-text"><i class="fas fa-star"></i> Cheapest Airline (on average): {cheapest_airline_str}</div>
        - <div class="icon-text"><i class="fas fa-exclamation-circle"></i> Most Expensive Airline (on average): {most_expensive_airline_str}</div>
        - <div class="icon-text"><i class="fas fa-city"></i> Number of Cities Covered: {num_cities}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Detailed Analysis")

        if 'filter_city' not in st.session_state: st.session_state.filter_city = "All"
        if 'filter_airline' not in st.session_state: st.session_state.filter_airline = "All"
        if 'filter_arrival' not in st.session_state: st.session_state.filter_arrival = "All"
        if 'analysis_type' not in st.session_state: st.session_state.analysis_type = "Average Price by Airline"

        def safe_index(options, value):
            try: return options.index(value)
            except ValueError: return 0

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            city_options = ["All"] + sorted(df_flights['From'].unique().tolist())
            st.session_state.filter_city = st.selectbox("Filter by Departure City", city_options, index=safe_index(city_options, st.session_state.filter_city))
        with col2:
            airline_options = ["All"] + sorted(df_flights['airline'].unique().tolist())
            st.session_state.filter_airline = st.selectbox("Filter by Airline", airline_options, index=safe_index(airline_options, st.session_state.filter_airline))
        with col3:
            arrival_options = ["All"] + sorted(df_flights['to'].unique().tolist())
            st.session_state.filter_arrival = st.selectbox("Filter by Arrival City", arrival_options, index=safe_index(arrival_options, st.session_state.filter_arrival))

        filtered_df = df_flights.copy()
        if st.session_state.filter_city != "All":
            filtered_df = filtered_df[filtered_df['From'] == st.session_state.filter_city]
        if st.session_state.filter_arrival != "All":
            filtered_df = filtered_df[filtered_df['to'] == st.session_state.filter_arrival]
        if st.session_state.filter_airline != "All":
            filtered_df = filtered_df[filtered_df['airline'] == st.session_state.filter_airline]

        analysis_options = [
            "Average Price by Airline", "Price Trend by Days Left", "Average Price by Number of Stops",
            "Average Price by Departure Time", "Price by City Pair", "Price by Class",
            "Busiest Routes", "Price Distribution"
        ]
        st.session_state.analysis_type = st.selectbox("Select Analysis Type", analysis_options, index=safe_index(analysis_options, st.session_state.analysis_type))

        if filtered_df.empty:
            st.warning("No data available for the selected filters. Please adjust your selections.")
        else:
            primary_plot_color = '#0A74DA'
            accent_plot_color = '#FF7F00'

            if st.session_state.analysis_type == "Average Price by Airline":
                df_airline = filtered_df.groupby('airline').agg({'price': 'mean'}).reset_index().sort_values('price')
                fig = px.bar(df_airline, x='airline', y='price', title="Average Price by Airline", color='airline', labels={'price': 'Average Price (INR)'}, color_discrete_sequence=px.colors.qualitative.Plotly)
                fig.update_layout(title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.analysis_type == "Price Trend by Days Left":
                df_days = filtered_df.groupby('days_left').agg({'price': 'mean'}).reset_index()
                fig = px.line(df_days, x='days_left', y='price', title="Price Trend by Days Left", labels={'price': 'Average Price (INR)'}, line_shape='spline', color_discrete_sequence=[primary_plot_color])
                fig.update_layout(title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.analysis_type == "Average Price by Number of Stops":
                df_stops = filtered_df.groupby('stops').agg({'price': 'mean'}).reset_index().sort_values('price')
                fig = px.bar(df_stops, x='stops', y='price', title="Average Price by Number of Stops", color='stops', labels={'price': 'Average Price (INR)'}, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.analysis_type == "Average Price by Departure Time":
                df_time = filtered_df.groupby('departure_time').agg({'price': 'mean'}).reset_index().sort_values('price')
                fig = px.bar(df_time, x='departure_time', y='price', title="Average Price by Departure Time", color='departure_time', labels={'price': 'Average Price (INR)'}, color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_layout(title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.analysis_type == "Price by City Pair":
                filtered_df['route'] = filtered_df['From'] + ' to ' + filtered_df['to']
                df_city = filtered_df.groupby('route').agg({'price': 'mean'}).reset_index().sort_values('price').head(10)
                fig = px.bar(df_city, x='route', y='price', title="Top 10 Cheapest Routes", color='route', labels={'price': 'Average Price (INR)'}, color_discrete_sequence=px.colors.qualitative.Vivid)
                fig.update_layout(title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.analysis_type == "Price by Class":
                df_class = filtered_df.groupby('Class').agg({'price': 'mean'}).reset_index()
                fig = px.bar(df_class, x='Class', y='price', title="Average Price by Class", color='Class', labels={'price': 'Average Price (INR)'}, color_discrete_map={'Economy': primary_plot_color, 'Business': accent_plot_color})
                fig.update_layout(title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.analysis_type == "Busiest Routes":
                filtered_df['route'] = filtered_df['From'] + ' to ' + filtered_df['to']
                df_routes = filtered_df.groupby('route').agg({'price': 'count'}).reset_index().sort_values('price', ascending=False).head(10)
                df_routes.columns = ['route', 'count']
                fig = px.bar(df_routes, x='route', y='count', title="Top 10 Busiest Routes", color='route', labels={'count': 'Number of Flights'}, color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_layout(title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
            
            elif st.session_state.analysis_type == "Price Distribution":
                fig = px.histogram(filtered_df, x='price', nbins=30, title="Price Distribution", labels={'price': 'Price (INR)'}, color_discrete_sequence=[primary_plot_color])
                fig.update_layout(bargap=0.1, title_font_family="Montserrat", font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)
# Predict Price page
elif page == "Predict Price for Business" and model is not None:
    st.header("Flight Price Predictor")
    st.markdown('<div class="icon-text" style="color: #0A74DA;"><i class="fas fa-search-dollar"></i> Predict flight prices with ease and confidence.</div>', unsafe_allow_html=True)
    
    if df_flights is None or df_flights.empty:
        st.error("Flight data is not available. Cannot make predictions.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            airline = st.selectbox("Airline", sorted(df_flights['airline'].unique()))
            departure = st.selectbox("Departure City", sorted(df_flights['From'].unique()))
            dep_time = st.selectbox("Departure Time", sorted(df_flights['departure_time'].unique()))
            stops = st.selectbox("Stops", ['Zero', 'One', 'Two or more'])
        
        with col2:
            arrival = st.selectbox("Arrival City", sorted(df_flights['to'].unique()))
            arr_time = st.selectbox("Arrival Time", sorted(df_flights['arrival_time'].unique()))
            flight_class = st.selectbox("Class", ['Economy', 'Business'])
            days_left = st.number_input("Days Before Flight", min_value=1, step=1, value=30, format="%d")
            st.markdown("**Flight Duration**")
            col_duration1, col_duration2 = st.columns([1, 1])
            with col_duration1:
                hours = st.number_input("Hours", min_value=0, step=1, value=0, format="%d")
            with col_duration2:
                minutes = st.number_input("Minutes", min_value=0, max_value=59, step=1, value=0, format="%d")
            duration = hours * 60 + minutes
        
        if duration < 30 and not (hours == 0 and minutes == 0): 
            st.markdown('<div class="stAlert"><i class="fas fa-exclamation-triangle"></i> Error: Flight duration must be at least 30 minutes.</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Quick Stats for Your Route")
        route_df = df_flights[(df_flights['From'] == departure) & (df_flights['to'] == arrival)]
        if not route_df.empty and departure != arrival:
            avg_route_price = route_df['price'].mean()
            num_flights_route = len(route_df)
            cheapest_airline_route_str = "N/A"
            cheapest_airline_route_group = route_df.groupby('airline')['price'].mean()
            if not cheapest_airline_route_group.empty:
                cheapest_airline_route = cheapest_airline_route_group.idxmin()
                cheapest_airline_price_val = cheapest_airline_route_group.min()
                cheapest_airline_route_str = f"{cheapest_airline_route} (₹{cheapest_airline_price_val:,.2f})"

            st.markdown(f"""
            - <div class="icon-text"><i class="fas fa-rupee-sign"></i> Average Price ({departure} to {arrival}): ₹{avg_route_price:,.2f}</div>
            - <div class="icon-text"><i class="fas fa-ticket-alt"></i> Number of Flights on this Route: {num_flights_route}</div>
            - <div class="icon-text"><i class="fas fa-star"></i> Cheapest Airline: {cheapest_airline_route_str}</div>
            """, unsafe_allow_html=True)
        elif departure == arrival:
             st.markdown('<div class="icon-text"><i class="fas fa-info-circle"></i> Departure and arrival cities are the same. Select different cities for stats.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="icon-text"><i class="fas fa-info-circle"></i> No direct flight data for this route in our dataset.</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("🔮 Predict Price"):
            if departure == arrival:
                st.markdown('<div class="stAlert"><i class="fas fa-exclamation-triangle"></i> Error: Departure and arrival cities cannot be the same.</div>', unsafe_allow_html=True)
            elif duration < 30 or (hours == 0 and minutes == 0):
                st.markdown('<div class="stAlert"><i class="fas fa-exclamation-triangle"></i> Error: Flight duration must be at least 30 minutes. Please specify hours and/or minutes.</div>', unsafe_allow_html=True)
            else:
                try:
                    # Ensure processed_df_flights is used for fitting encoders if it's the one used for model training
                    # If the original df_flights was used for training, then fit on that.
                    # Assuming processed_df_flights is the correct one as it contains encoded columns.
                    if processed_df_flights.empty:
                         st.error("Processed flight data is not available for prediction encoding.")
                         st.stop()

                    le_airline = LabelEncoder().fit(processed_df_flights['airline'])
                    le_from = LabelEncoder().fit(processed_df_flights['From'])
                    le_to = LabelEncoder().fit(processed_df_flights['to'])
                    le_dep_time = LabelEncoder().fit(processed_df_flights['departure_time'])
                    le_arr_time = LabelEncoder().fit(processed_df_flights['arrival_time'])

                    airline_enc = le_airline.transform([airline])[0]
                    dep_enc = le_from.transform([departure])[0]
                    arr_enc = le_to.transform([arrival])[0]
                    dep_time_enc = le_dep_time.transform([dep_time])[0]
                    arr_time_enc = le_arr_time.transform([arr_time])[0]
                    
                    stops_map = {'Zero': 0, 'One': 1, 'Two or more': 2}
                    stops_enc = stops_map[stops]
                    class_map = {'Economy': 0, 'Business': 1}
                    class_enc = class_map[flight_class]
                    
                    input_data = np.array([[
                        duration, days_left, airline_enc, 
                        dep_time_enc, dep_enc, arr_time_enc,
                        arr_enc, class_enc, stops_enc
                    ]])
                    
                    prediction = model.predict(input_data)[0]
                    st.markdown(f'<div class="stSuccess"><i class="fas fa-check-circle"></i> Predicted Price: ₹{prediction:,.2f}</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("Contextual Insights")
                    overall_avg = df_flights['price'].mean()
                    st.markdown(f'<div class="icon-text"><i class="fas fa-globe-asia"></i> Overall Average Flight Price (All Routes): ₹{overall_avg:,.2f}</div>', unsafe_allow_html=True)
                    airline_avg = df_flights[df_flights['airline'] == airline]['price'].mean()
                    st.markdown(f'<div class="icon-text"><i class="fas fa-plane"></i> Average Price for {airline}: ₹{airline_avg:,.2f}</div>', unsafe_allow_html=True)
                    class_avg = df_flights[df_flights['Class'] == flight_class]['price'].mean()
                    st.markdown(f'<div class="icon-text"><i class="fas fa-chair"></i> Average Price for {flight_class} Class: ₹{class_avg:,.2f}</div>', unsafe_allow_html=True)
                    
                    stops_df_temp = df_flights[df_flights['stops'] == stops]
                    stops_avg = stops_df_temp['price'].mean() if not stops_df_temp.empty else overall_avg
                    st.markdown(f'<div class="icon-text"><i class="fas fa-map-signs"></i> Average Price for {stops} Stops: ₹{stops_avg:,.2f} (compare with overall)</div>', unsafe_allow_html=True)
                    days_left_df_temp = df_flights[df_flights['days_left'] == days_left]
                    days_left_avg = days_left_df_temp['price'].mean() if not days_left_df_temp.empty else overall_avg
                    st.markdown(f'<div class="icon-text"><i class="fas fa-calendar-alt"></i> Average Price for Booking {days_left} Days Left: ₹{days_left_avg:,.2f} (compare with overall)</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown(f'<div class="stAlert"><i class="fas fa-exclamation-triangle"></i> Prediction failed: {str(e)}. Please check inputs or model.</div>', unsafe_allow_html=True)

# Travel Corner page
elif page == "Traveler Corner":
    st.header("Traveler Corner")
    st.markdown('<div class="icon-text" style="color: #0A74DA;"><i class="fas fa-suitcase-rolling"></i> Get personalized travel tips for your  trips.</div>', unsafe_allow_html=True)
    
    if df_flights is None or df_flights.empty:
        st.error("Flight data is not available. Cannot provide travel tips.")
    else:
        budget = st.number_input("Enter your budget (INR)", min_value=1000, step=100, value=5000, format="%d")
        
        departure_options_tc = sorted(df_flights['From'].dropna().unique().tolist())
        departure_tc = st.selectbox("Select Departure City", departure_options_tc, key="tc_dep")
        
        arrival_options_tc = sorted(df_flights["to"].dropna().unique().tolist())
        arrival_tc = st.selectbox("Select Arrival City", arrival_options_tc, key="tc_arr")
        
        if st.button("💡 Get Travel Tips"):
            if departure_tc == arrival_tc:
                st.markdown('<div class="stAlert"><i class="fas fa-exclamation-triangle"></i> Error: Departure and arrival cities cannot be the same.</div>', unsafe_allow_html=True)
            else:
                route_df_tc = df_flights[(df_flights['From'] == departure_tc) & (df_flights['to'] == arrival_tc)]
                if route_df_tc.empty:
                    st.markdown('<div class="icon-text"><i class="fas fa-info-circle"></i> No direct flight data available for this route. Cannot generate specific tips.</div>', unsafe_allow_html=True)
                else:
                    st.subheader(f"Tips for {departure_tc} to {arrival_tc} within ₹{budget:,.0f} budget")
                    
                    budget_flights = route_df_tc[route_df_tc['price'] <= budget]
                    
                    if budget_flights.empty:
                        st.markdown(f'<div class="icon-text"><i class="fas fa-sad-tear"></i> No flights found within your budget of ₹{budget:,.0f} for this route. Consider increasing your budget or checking other routes.</div>', unsafe_allow_html=True)
                        cheapest_overall_on_route = route_df_tc['price'].min()
                        st.markdown(f'<div class="icon-text"><i class="fas fa-info-circle"></i> The cheapest flight on this route currently costs ₹{cheapest_overall_on_route:,.2f}.</div>', unsafe_allow_html=True)
                    else:
                        best_airline_budget_str = "N/A"
                        best_airline_budget_group = budget_flights.groupby('airline')['price'].mean()
                        if not best_airline_budget_group.empty:
                            best_airline_budget = best_airline_budget_group.idxmin()
                            avg_price_best_airline_budget = best_airline_budget_group.min()
                            best_airline_budget_str = f"{best_airline_budget} (Average Price: ₹{avg_price_best_airline_budget:,.2f})"

                        best_time_budget_str = "N/A"
                        best_time_budget_group = budget_flights.groupby('departure_time')['price'].mean()
                        if not best_time_budget_group.empty:
                            best_time_budget = best_time_budget_group.idxmin()
                            avg_price_best_time_budget = best_time_budget_group.min()
                            best_time_budget_str = f"{best_time_budget} (Average Price: ₹{avg_price_best_time_budget:,.2f})"
                        
                        optimal_days_budget_str = "N/A"
                        optimal_days_budget_group = budget_flights.groupby('days_left')['price'].mean()
                        if not optimal_days_budget_group.empty:
                            optimal_days_budget = optimal_days_budget_group.idxmin()
                            avg_price_optimal_days_budget = optimal_days_budget_group.min()
                            optimal_days_budget_str = f"{optimal_days_budget} days in advance (Average Price: ₹{avg_price_optimal_days_budget:,.2f})"

                        cheapest_flight_budget = budget_flights['price'].min()
                        
                        st.markdown("### Smart Travel Recommendations:")
                        st.markdown(f"""<div class="insights-box">
                        - <div class="icon-text"><i class="fas fa-plane-departure"></i> **Best Airline (within budget):** {best_airline_budget_str}</div>
                        - <div class="icon-text"><i class="fas fa-clock"></i> **Best Departure Time (within budget):** {best_time_budget_str}</div>
                        - <div class="icon-text"><i class="fas fa-calendar-check"></i> **Optimal Days to Book (within budget):** {optimal_days_budget_str}</div>
                        - <div class="icon-text"><i class="fas fa-tags"></i> **Cheapest Flight Found (within budget):** ₹{cheapest_flight_budget:,.2f}</div>
                        </div>""", unsafe_allow_html=True)

                        st.markdown("### General Savings Tips:")
                        st.markdown("""<div class="insights-box">
                        - <div class="icon-text"><i class="fas fa-user-friends"></i> Consider flying during off-peak hours or mid-week for potentially lower fares.</div>
                        - <div class="icon-text"><i class="fas fa-briefcase"></i> If possible, travel light to avoid extra baggage fees, especially on budget airlines.</div>
                        - <div class="icon-text"><i class="far fa-calendar-alt"></i> Booking further in advance often yields better prices, but also check for last-minute deals if your schedule is flexible.</div>
                        </div>""", unsafe_allow_html=True)

# Ensure model and df_flights are loaded before trying to access them
if model is None or (df_flights is None or df_flights.empty):
    st.error("Critical error: Model or flight data could not be loaded. Application cannot run fully.")
    # st.stop() # Commenting out stop to allow other parts of UI to render if possible

