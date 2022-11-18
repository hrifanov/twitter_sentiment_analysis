import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys

INPUT_FILE=''
TITLE=''


# word cloud with most frequent words
def show_wordcloud(df):
    all_words = ' '.join([tweets for tweets in df['clean_text']])
    word_cloud = WordCloud(width=800, height=500, random_state=1337, max_font_size=100).generate(all_words)
    plt.figure(figsize=(20,10))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
 
# barchart for sentiment distribution (positive, neutral, negative)
def show_barchard(data, title):
    plot = data['sentiment'].value_counts().plot(kind='bar', figsize=(10, 5), title=title + ' Twitter Sentiment Distribution')
    plot.set_xlabel("Sentiment")
    plot.set_ylabel("Frequency")
    plt.show()

# piechart for sentiment distribution (positive, neutral, negative)
def show_piechard(data, title):
    plot = data['sentiment'].value_counts().plot.pie(y='sentiment', figsize=(10, 10), title=title + ' Twitter Sentiment Distribution')
    plt.show()

# donut chart for weighed distribution of positive versus negative polarity + mean and absolute weighed polarity
def show_donutchart(data, title):
    positive = data[data['weighed_polarity'] > 0]['weighed_polarity'].sum()
    negative = data[data['weighed_polarity'] < 0]['weighed_polarity'].sum() * (-1)

    colors = ['#65c368', '#FF0000']

    gaps = (0.02, 0.02)

    fig = plt.figure(figsize=(10, 10))

    plt.pie([positive, negative], colors=colors, labels=['Positive', 'Negative'],
            autopct='%1.1f%%', pctdistance=0.85,
            explode=gaps)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')

    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('[TOTAL WEIGHED POLARITY: ' + str(round((positive - negative), 3)) + ']\n' +
            '[MEAN WEIGHED POLARITY: ' + str(round((positive - negative) / data.shape[0], 3)) + ']\n\n' +
            title + ' Weighed Polarity Distribution ')
    plt.show()

# scatter plot of polarity and subjectivity
def show_scatterplot(data, title):
    plt.figure(figsize=(14, 7))
    for i in range(0, data.shape[0]):
        plt.scatter(data['polarity'][i], data['subjectivity'][i], color='blue')

    plt.grid()
    plt.title(title + ' Twitter Polarity vs. Subjectivity')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')

    plt.show()

def main(input_file, title):
    # load dataset
    df = pd.read_csv(input_file)

    #shot visualizations
    show_wordcloud(df)
    show_barchard(df, title)
    show_piechard(df, title)
    show_donutchart(df, title)
    show_scatterplot(df, title)

main(INPUT_FILE, TITLE)
