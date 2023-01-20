from surprise import Dataset, KNNBasic, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

# Load the movielens-100k dataset
print("Setting Up Datasets...")
data = Dataset.load_builtin("ml-100k")
data_df = pd.DataFrame(data.__dict__['raw_ratings'], columns=['user_id','item_id','rating','timestamp'])
data_df_c = data_df.copy()
data_df = data_df[data_df["user_id"] == '22']
datatoexclude = data_df["item_id"].values
items_to_rate = data_df_c[~data_df_c["item_id"].isin(datatoexclude)]["item_id"].values
items_to_rate = list(set(items_to_rate))
items_to_check = data_df_c[data_df_c["item_id"].isin(datatoexclude)]["item_id"].values
items_to_check = list(set(items_to_rate))

print("Establishing best algos...")
param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
gs = GridSearchCV(SVD,param_grid, measures=["rmse", "mae"], cv=3)

# Build an algorithm, and train it.
print("Build and Train Algo...")
gs.fit(data)
algo = gs.best_estimator["rmse"]
algo.fit(data.build_full_trainset())

trainset, testset = train_test_split(data, test_size=0.25)
pred = algo.test(testset)
print("SVD Accuracy: ",accuracy.rmse(pred))

def hideThis():
    res = pd.DataFrame(columns = ["item","est_score"])
    for el in items_to_rate:
        uid = "22"
        iid = str(el)
        r = algo.predict(uid,iid)
        d = {

            "item": [r[1]],
            "est_score": [r[-2]]
        
        }


        res = pd.concat([res,pd.DataFrame(d)])

    print(res.sort_values(by = "est_score",ascending=False).head())

print("Generate Results...")
data_test = data_df_c[data_df_c["user_id"] == "22"].head(100)
print(algo.predict(22,227))
print(data_test.shape)
for ela in data_test.iterrows():
    val = algo.predict(22,str(ela[1]["item_id"]))
    
    data_test.at[ela[0],"Pred Rating"] = val[-2]


data_test.plot(x = "item_id", y = ["rating","Pred Rating"]).get_figure().savefig('line.png')
data_test.plot(kind = "box", y = ["rating","Pred Rating"]).get_figure().savefig('box.png')



