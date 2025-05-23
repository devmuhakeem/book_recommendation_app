import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load sample data from notebook
ratings = pd.DataFrame({
    'user_id': pd.np.random.randint(1, 21, 100),
    'book_id': pd.np.random.randint(1, 21, 100),
    'rating': pd.np.random.randint(1, 6, 100)
})
books = pd.DataFrame({
    'book_id': range(1, 21),
    'title': [f"Book {i}" for i in range(1, 21)],
    'author': [f"Author {i}" for i in range(1, 21)]
})

user_item_matrix = ratings.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_books(user_id, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    weighted_scores = user_item_matrix.T.dot(similar_users)
    user_rated_books = user_item_matrix.loc[user_id]
    weighted_scores = weighted_scores.drop(user_rated_books[user_rated_books > 0].index, errors='ignore')

    top_books = weighted_scores.sort_values(ascending=False).head(num_recommendations).index
    return books[books['book_id'].isin(top_books)][['title', 'author']].to_dict(orient='records')
