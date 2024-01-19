import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import streamlit as st

# Load the dataset
file_path = "final_ratings"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Create a Surprise Dataset
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['User-ID', 'ISBN', 'Book-Rating']], reader)

# Build the train set
trainset = data.build_full_trainset()

# Build the SVD model
svd_model = SVD()
svd_model.fit(trainset)

# Streamlit app
st.title("Book Recommendation App")

# Get book ratings based on input book title
input_book_title = st.text_input("Enter a book title:", "The Da Vinci Code")  # Default book title for example
book_ratings = df[df['Book-Title'] == input_book_title][['User-ID', 'ISBN', 'Book-Rating']]

if not book_ratings.empty:
    # Train the SVD model on the entire dataset
    svd_model.fit(trainset)

    # Predict ratings for other books based on users who rated the input book
    user_ids = book_ratings['User-ID'].tolist()
    other_books = df[df['User-ID'].isin(user_ids) & (df['Book-Title'] != input_book_title)]['ISBN'].unique()
    predictions = [svd_model.predict(user_id, book) for user_id in user_ids for book in other_books]

    # Sort predictions in descending order of estimated rating
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    st.subheader("Top 5 Recommended Books:")

    # Display top 5 recommended books
    for i, prediction in enumerate(sorted_predictions[:5]):
        book_info = df[df['ISBN'] == prediction.iid][['Book-Title', 'Book-Author']].iloc[0]
        st.write(f"{i + 1}. {book_info['Book-Title']} by {book_info['Book-Author']} (Predicted Rating: {prediction.est:.2f})")
else:
    st.warning("Book not found. Please enter a valid book title.")
