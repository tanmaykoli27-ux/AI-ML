# AI-Powered Smart Tourism Recommender
# Student Name: Tanmay Koli

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of users and places they rated
data = {
    "User": ["U1","U1","U1","U2","U2","U2","U3","U3","U3","U4","U4"],
    
    "Place": [
        "Jaipur Fort",
        "Udaipur Palace",
        "Varanasi Ghats",
        "Jaipur Fort",
        "Shimla Homestay",
        "Manali Cottage",
        "Udaipur Palace",
        "Varanasi Ghats",
        "Manali Cottage",
        "Shimla Homestay",
        "Manali Cottage"
    ],
    
    "Rating": [5,4,4,5,4,5,4,5,4,5,4]
}

df = pd.DataFrame(data)

# Convert into user-item matrix
matrix = df.pivot_table(index="User", columns="Place", values="Rating").fillna(0)

print("User-Item Matrix:\n")
print(matrix)

# Calculate similarity between items
similarity = cosine_similarity(matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=matrix.columns,
    columns=matrix.columns
)

print("\nItem Similarity Matrix:\n")
print(similarity_df)

# Function to recommend places
def recommend(place_name, top_n=3):
    
    scores = similarity_df[place_name].sort_values(ascending=False)
    
    scores = scores.drop(place_name)
    
    return scores.head(top_n)

# Example recommendation
print("\nRecommended places if user liked Jaipur Fort:\n")

print(recommend("Jaipur Fort"))
