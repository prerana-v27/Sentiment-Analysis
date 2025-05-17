# Sentiment Analysis Project

## Overview

This project is a Sentiment Analysis application that uses Natural Language Processing (NLP) techniques to analyze text reviews and determine their sentiment (positive or negative). It is built using a pre-trained BERT model and features a user-friendly interface powered by Streamlit.

## Features

* Analyze a single text review for sentiment (Positive or Negative).
* Analyze a CSV file containing multiple reviews for sentiment classification.
* Visualize sentiment distribution of reviews using matplotlib and seaborn.

## Project Structure

* `app.py`: The main Streamlit app file.
* `notebooks/`: Jupyter notebooks for data preprocessing and model training.
* `Report`: Senyiment Analysis Report (pdf).

## Tools and Libraries Used

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit
* Pandas, NumPy
* Matplotlib, Seaborn

## Dataset

The dataset used in this project is the IMDB Movie Reviews dataset from Kaggle.
Kaggle IMDB Dataset Link

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/prerana-v27/Sentiment-Analysis/edit/main/README.md
   cd Sentiment-Analysis-Project
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Pre-trained Model:**
   Make sure the fine-tuned BERT model is saved in the `model/` directory.

4. **Run the Streamlit Application:**

   ```bash
   streamlit run sentiment_analysis_app.py
   ```

## Usage

* To analyze a single review, enter the text in the app and click "Analyze".
* To analyze a dataset, upload a CSV file containing a column named 'review'.
* The app will display the sentiment for each review and visualize the sentiment distribution.

## License

This project is open-source and available under the MIT License.
