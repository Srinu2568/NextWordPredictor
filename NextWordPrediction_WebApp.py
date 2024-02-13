import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import plotly.graph_objects as go
from collections import Counter
from tensorflow.keras.optimizers import Adam
import pandas as pd
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Next Word Prediction App", page_icon="✨")

filtered_sentences = []

with open('PreprocssedData - Sheet1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        filtered_sentences.append(row)

col_name = ["Sentences"]
df_filtered = pd.DataFrame(filtered_sentences, columns=col_name)
df_filtered = df_filtered[df_filtered["Sentences"] != "Sentences"]

model = load_model('./my_model.h5', compile=False)

# Manually load the custom optimizer
custom_optimizer = Adam(clipvalue=0.5)  # Adjust parameters accordingly

# Compile the model with the loaded custom optimizer
model.compile(optimizer=custom_optimizer, loss='your_loss_function', metrics=['accuracy'])

# Use Streamlit caching for computationally expensive functions
@st.cache(allow_output_mutation=True)
def make_tokenizer():
    sentences = df_filtered['Sentences'].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer

tokenizer = make_tokenizer()

@st.cache(allow_output_mutation=True)
def create_counter():
    sentences = df_filtered['Sentences'].tolist()
    words = [word_tokenize(sentence) for sentence in sentences]
    words_without_stopwords = [word for line in words for word in line if word.lower() not in stop_words]
    words_freq = Counter(words_without_stopwords)
    return words_freq

stop_words = set(stopwords.words('english'))

words_freq = create_counter()

def show_wordcloud():
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(words_freq))
    st.image(wordcloud.to_array())

def show_bar_chart():
    total_stopwords, total_other_words = zip(*[count_words(sentence) for sentence in df_filtered['Sentences']])
    st.bar_chart({'Stopwords': [sum(total_stopwords)], 'Other Words': [sum(total_other_words)]})

def count_words(sentence):
    words = word_tokenize(sentence)
    num_stopwords = len([word for word in words if word.lower() in stop_words])
    num_other_words = len(words) - num_stopwords
    return num_stopwords, num_other_words

def show_histogram():
    word_lengths = df_filtered['Sentences'].apply(lambda x: len(x.split()))

    fig = go.Figure(data=[go.Histogram(x=word_lengths, xbins=dict(start=0, end=100),
                                       marker_color='#3498db', marker_line_color='black', marker_line_width=1.5)])
    fig.update_layout(title='Distribution of Word Lengths in Filtered Sentences',
                      xaxis_title='Word Length', yaxis_title='Frequency')
    
    st.plotly_chart(fig)

def show_line_chart():
    top_words = words_freq.most_common(10)  # Display top 10 words
    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    st.line_chart(top_words_df.set_index('Word'))

def show_top_words_table():
    top_words = words_freq.most_common(10)  # Display top 10 words
    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    st.table(top_words_df)

def predict_next_word(input_words, loaded_filtered_model, tokenizer, max_sequence_length):
    input_sequence = tokenizer.texts_to_sequences([input_words])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_sequence_length - 1, padding='pre')
    predicted_probabilities = loaded_filtered_model.predict(input_sequence, verbose=0).flatten()

    top_n = 3
    top_indices = predicted_probabilities.argsort()[-top_n:][::-1]
    top_words = [tokenizer.index_word.get(idx, "<Unknown>") for idx in top_indices]
    return top_words

def home_page():
    st.title("Welcome to Next Word Prediction Project")
    st.write(
        "This project utilizes LSTM to predict the next word in a given sentence. "
        "The model is trained on a dataset, and it suggests the most probable next words based on the input."
    )

    if st.checkbox("Show Dataset"):
        load_dataset()

def visualization_page():
    st.title("Data Visualizations")

    st.subheader("Wordcloud")
    show_wordcloud()

    st.subheader("Bar Chart")
    show_bar_chart()

    st.subheader("Histogram")
    show_histogram()

    st.subheader("Line Chart")
    show_line_chart()

    st.subheader("Top Words Table")
    show_top_words_table()

def next_word_input_page():
    st.title("Next Word Prediction App")

def load_dataset():
    st.subheader("Displaying Dataset:")
    st.dataframe(df_filtered)

def main():
    home_page()
    st.subheader("Try the model below:")
    placeholder = st.empty()

    user_input = st.text_input("Enter a sentence:", "")

    if user_input:
        predicted_words = predict_next_word(user_input, model, tokenizer, 724)
        placeholder.text(f"Predicted Next Words: {', '.join(predicted_words)}")

    visualization_page()

if __name__ == '__main__':
    main()