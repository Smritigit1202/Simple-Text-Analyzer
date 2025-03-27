# Text Analysis Tool

## Overview
This Python script performs text summarization, sentiment analysis, and word cloud generation using NLP libraries such as NLTK, TextBlob, and WordCloud. It processes input text to extract key sentences, determine sentiment polarity, and visualize word frequency.

## Features
- **Text Summarization**: Extracts important sentences based on word frequency.
- **Sentiment Analysis**: Identifies sentiment as Positive, Negative, or Neutral.
- **Word Cloud Generation**: Creates a visual representation of word frequency.

## Dependencies
Ensure the following Python libraries are installed before running the script:
```bash
pip install textblob wordcloud nltk matplotlib
```

## Usage
1. Run the script with a sample text input.
2. The program will generate a summary, sentiment score, and word cloud image.
3. The word cloud will be saved and available for download.

## Example Output
**Word Cloud:**  
![wordcloud (1)](https://github.com/user-attachments/assets/9346ad69-01fa-4200-9b78-a07ebb9645fa)


## NLTK Data Download
Ensure necessary NLTK datasets are downloaded before execution:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
```


