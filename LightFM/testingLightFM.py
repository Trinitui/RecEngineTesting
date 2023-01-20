from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
import numpy as np
import pandas as pd

# REQUIRES NUMPY 1.23.4 !!!

# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.
data = fetch_movielens()

# Instantiate and train the model
model = LightFM(loss='warp-kos') 
model.fit(data['train'], epochs=30, num_threads=2)

# Evaluate the trained model

print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())


def sample_recommendation(model, data, user_ids):

    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendation(model, data, [22])

a = pd.DataFrame(data)
print(a.head())