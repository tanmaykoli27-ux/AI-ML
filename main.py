# AI-Powered Smart Tourism Recommender & Demand Optimizer
# Student: Tanmay Koli

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression


# -------------------------------
# PART 1: TOURISM RECOMMENDER
# -------------------------------

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

matrix = df.pivot_table(index="User", columns="Place", values="Rating").fillna(0)

similarity = cosine_similarity(matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=matrix.columns,
    columns=matrix.columns
)

def recommend(place_name, top_n=3):

    scores = similarity_df[place_name].sort_values(ascending=False)

    scores = scores.drop(place_name)

    return scores.head(top_n)


# -------------------------------
# PART 2: DEMAND OPTIMIZER
# -------------------------------

def predict_demand():

    # month vs booking data
    months = np.array([1,2,3,4,5,6]).reshape(-1,1)

    bookings = np.array([120,150,170,200,220,260])

    model = LinearRegression()

    model.fit(months, bookings)

    future_month = np.array([[7]])

    prediction = model.predict(future_month)

    return int(prediction[0])


# -------------------------------
# MAIN PROGRAM
# -------------------------------

print("\n--- TOURISM RECOMMENDER ---")

place = input("Enter place you liked: ")

print("\nRecommended places:\n")

print(recommend(place))


print("\n--- DEMAND OPTIMIZER ---")

result = predict_demand()

print("\nExpected bookings next month:", result)
