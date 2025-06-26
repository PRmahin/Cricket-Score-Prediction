import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

# Load the trained model pipeline
pipe = pickle.load(open('pipeline_nn.pkl', 'rb'))

# Define the list of teams and venues
teams = ['England', 'Pakistan', 'Sri Lanka', 'Australia', 'South Africa',
         'New Zealand', 'India', 'Zimbabwe', 'West Indies', 'Ireland',
         'Scotland', 'Kenya', 'Bangladesh', 'Afghanistan']

venues = ['New Wanderers Stadium', 'Sophia Gardens', 'Providence Stadium',
          'Kennington Oval', 'Sydney Cricket Ground', 'Edgbaston',
          'Brisbane Cricket Ground, Woolloongabba', 'Eden Park',
          'Melbourne Cricket Ground', "Queen's Park Oval, Port of Spain",
          'Shere Bangla National Stadium', 'Bellerive Oval',
          'Sheikh Zayed Stadium', 'Newlands', 'The Rose Bowl',
          'Riverside Ground', 'Saxton Oval', 'Kingsmead',
          'Warner Park, Basseterre', "National Cricket Stadium, St George's",
          'Trent Bridge', 'Western Australia Cricket Association Ground',
          'Punjab Cricket Association Stadium, Mohali',
          'Kensington Oval, Bridgetown', 'SuperSport Park',
          'Rangiri Dambulla International Stadium', 'Nehru Stadium',
          'R Premadasa Stadium', 'MA Chidambaram Stadium, Chepauk',
          'Adelaide Oval', 'Vidarbha Cricket Association Stadium, Jamtha',
          'Sir Vivian Richards Stadium, North Sound', 'Feroz Shah Kotla',
          'Eden Gardens', 'Sharjah Cricket Stadium', 'Sabina Park, Kingston',
          'Dubai International Cricket Stadium', 'University Oval',
          'Kinrara Academy Oval', 'Westpac Stadium', 'Seddon Park',
          'Headingley', 'Arnos Vale Ground, Kingstown',
          'Civil Service Cricket Club, Stormont', 'Old Trafford',
          'National Stadium', 'Pallekele International Cricket Stadium',
          "St George's Park", "Lord's", 'Sawai Mansingh Stadium',
          'Multan Cricket Stadium', 'Harare Sports Club', 'McLean Park',
          'Khan Shaheb Osman Ali Stadium', 'Hagley Oval', 'Gaddafi Stadium',
          'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa',
          'Zahur Ahmed Chowdhury Stadium', 'M Chinnaswamy Stadium',
          'Queens Sports Club', 'Wankhede Stadium',
          'Beausejour Stadium, Gros Islet', 'Manuka Oval',
          'Sardar Patel Stadium, Motera', 'Clontarf Cricket Club Ground',
          'Willowmoore Park']

# Streamlit app title
st.title('Cricket Score Predictor')

# Layout with two columns for input
col1, col2 = st.columns(2)

# Select batting team and bowling team
with col1:
    bat_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowl_team = st.selectbox('Select bowling team', sorted(teams))

# Select the venue
venue = st.selectbox('Select Venue', sorted(venues))

# Input fields for current score, overs, wickets, wickets in last 5 overs, and runs in last 5 overs
runs = st.number_input('Current Score')
overs = st.number_input('Overs done (works for over > 5)')
wickets = st.number_input('Wickets out')
wickets_last_5 = st.number_input('Wickets in last five overs')
last_five = st.number_input('Runs scored in last 5 overs')

# Prediction button
if st.button('Predict Score'):
    # Calculate remaining overs, weight over, wickets left, weight wicket, merge weight, and balls left
    remaining_overs = 50 - overs
    weight_over = remaining_overs / 49.6
    wickets_left = 10 - wickets
    weight_wicket = wickets_left / 10
    merge_weight = (remaining_overs * weight_over) + (wickets_left * weight_wicket)
    balls_left = 50 - (overs * 6)
    runrate = runs / overs

    # Create a DataFrame with the input data
    input_df = pd.DataFrame({
        'bat_team': [bat_team], 'bowl_team': [bowl_team],
        'venue': [venue], 'overs': [overs], 'runs': [runs],
        'wickets': [wickets], 'runrate': [runrate], 'runs_last_5': [last_five],
        'wickets_last_5': [wickets_last_5]
    })

    # Predict the score
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))

    # Display additional stats
    st.title('Main Stats')
    st.text("Balls Left: " + str(balls_left))
    st.text("Wickets Left: " + str(wickets_left))
    st.text("Run Rate: " + str(runrate))

# Flask integration
if __name__ == '__main__':
    import os
    from flask import Flask, send_file

    app = Flask(__name__)

    @app.route('/')
    def main():
        return st.script_runner.__script__.get_code()

    @app.route('/download_app', methods=['GET'])
    def download_app():
        file_path = 'path_to_your_script.py'
        return send_file(file_path, as_attachment=True)

    if 'ON_HEROKU' in os.environ:
        app.run(port=int(os.environ.get('PORT', 33507)))
    else:
        app.run(debug=True)