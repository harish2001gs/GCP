import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
from dotenv import load_dotenv
import textwrap
import json

def analyze_reviews(file_path, api_key):
    # Load data
    df = pd.read_excel(file_path)
    df_cleaned = df.dropna()

    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Define sentiment scoring function
    def get_sentiment_scores(text):
        scores = sia.polarity_scores(text)
        return scores

    # Create a DataFrame for reviews
    df1 = pd.DataFrame()
    df1['reviews'] = df_cleaned['review_text']
    df1['Name']=df_cleaned['Name']

    # Apply sentiment analysis
    df1['SentimentScores'] = df1['reviews'].apply(get_sentiment_scores)
    df1['Compound'] = df1['SentimentScores'].apply(lambda x: x['compound'])
    
    # Count sentiments
    positive_count = df1[df1['Compound'] > 0]['Compound'].count()
    negative_count = df1[df1['Compound'] < 0]['Compound'].count()
    neutral_count = df1[df1['Compound'] == 0]['Compound'].count()
    
    # Print sentiment counts
    print("Positive sentiment:", positive_count)
    print("Negative sentiment:", negative_count)
    print("Neutral sentiment:", neutral_count)

    # Prepare data for JSON
    new = df1[['Name', 'reviews', 'Compound']]
    df_json_zero = new.to_json(orient='records')

    # Configure Google Generative AI
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")

    # Prompt for sentiment rating
    prompt_zero = f"""You are an expert in sentiment analysis, skilled at rating customer reviews based on their sentiment scores.
    Your task is to assign a sentiment rating to the customer reviews provided between three backticks. The ratings should range from 0 to 10, where:
    - 0 indicates a highly negative sentiment,
    - 10 indicates a highly positive sentiment.

    The customer reviews are provided in JSON format, and your output should only include the JSON code with the updated 'sentiment_rating' values. Please do not change the JSON code format.
    {df_json_zero}
    """

    # Generate content
    response_zero = model.generate_content(prompt_zero)
    response_data_zero = pd.DataFrame(json.loads(response_zero.text.strip("")))

    # Add manager rating and final rating
    response_data_zero['manager_rating'] = df_cleaned['manager_rating'].values
    response_data_zero['final_rating'] = response_data_zero['sentiment_rating'] + response_data_zero['manager_rating']
    print(response_data_zero)
    return response_data_zero


df_analyzed = analyze_reviews("Untitled spreadsheet.xlsx","AIzaSyDO6cArWtL109TvF9pxT7V5tccTCsbTJmA")
print(df_analyzed)

@app.route('/', methods=['GET', 'POST'])
def index():
    review_rating = None
    name = None

    if request.method == 'POST':
        name = request.form['name']
        print(name)
        # Check if the name exists in the DataFrame
        if name in df_analyzed['Name'].values: 
            print(name) # Replace 'name_column' with the actual column name
            review_rating = df_analyzed.loc[df_analyzed['Name'] == name, 'final_rating'].values[0]  # Replace 'sentiment_rating' with the actual column name
        else:
            review_rating = "Name not found."

    return render_template('index.html', review_rating=review_rating, name=name)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
