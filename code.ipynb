# Install dependencies (if needed)
!pip install textblob wordcloud nltk matplotlib

# Import required libraries
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.probability import FreqDist
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('punkt_tab') # Download the punkt_tab data package

# Text summarization function
def text_summarize(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    frequency_dist = FreqDist(words)
    max_freq = max(frequency_dist.values())
    sentence_scores = {}

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency_dist.keys():
                if sentence in sentence_scores:
                    sentence_scores[sentence] += frequency_dist[word] / max_freq
                else:
                    sentence_scores[sentence] = frequency_dist[word] / max_freq

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:2]
    summary = TreebankWordDetokenizer().detokenize(summary_sentences)
    return summary

# Sentiment analysis function
def sentiment_analysis(text):
    analysis = TextBlob(text)
    sentiment = SentimentIntensityAnalyzer()
    sent = sentiment.polarity_scores(text)
    result = ""
    if analysis.sentiment.polarity > 0:
        result = "Positive"
    elif analysis.sentiment.polarity < 0:
        result = "Negative"
    else:
        result = "Neutral"
    result = result + " : " + str(sent)
    return result

# Word cloud function
def word_cloud(text, filename):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")

    # Save file in Colab's default directory
    file = f'/content/{filename}'
    plt.savefig(file)
    plt.close()
    
    return file

# Example usage
text = "Hi, I am Smriti Aggarwal. I am a software engineer with a passion for artificial intelligence and data science. I graduated with a BTech. My expertise lies in machine learning, natural language processing, and cloud computing.I am always eager to learn and collaborate on innovative projects. Feel free to connect with me for discussions on AI, tech trends, or exciting opportunities!"
print("Summary:", text_summarize(text))
print("Sentiment:", sentiment_analysis(text))

# Generate and download the word cloud
word_cloud_file = word_cloud(text, "wordcloud.png")

from google.colab import files
files.download(word_cloud_file)  # Download the file
