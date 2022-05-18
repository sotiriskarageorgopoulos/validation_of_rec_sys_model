import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

def calc_rmse(actual_data_points, predicted_data_points, number_of_points):
    '''
        Calculates the RMSE(Root Mean Square Error) of true and predicted data points.
    '''
    sum = 0
    for i in range(number_of_points):
        sum += math.pow(actual_data_points[i] - predicted_data_points[i],2)
    return float("{:.3f}".format(math.sqrt(sum / number_of_points)))

logging.basicConfig(level = logging.INFO)
rating_matrix = np.genfromtxt('./data/user-shows.txt',delimiter=' ')
logging.info('The ratings matrix is ready...')
item_item_rec_matrix = np.genfromtxt('./data/item-item-rec-matrix.txt',delimiter=' ')
logging.info('The item-item collaborative filtering recommendation matrix is ready...')
user_user_rec_matrix = np.genfromtxt('./data/user-user-rec-matrix.txt',delimiter=' ')
logging.info('The user-user collaborative filtering recommendation matrix is ready...')

ratings = rating_matrix[7987:,0:111]
item_item_recommendations = item_item_rec_matrix[7987:,0:111]
user_user_recommendations = user_user_rec_matrix[7987:,0:111]

ii_actual_ratings = []
uu_actual_ratings = []

rmse_for_each_k = {
    "k": [],
    "item_item_rmse": [],
    "user_user_rmse": []
}

for k in range(1,31):
    rmse_for_each_k["k"].append(k)
    logging.info(f"The k is: {k}")
    for user_idx in range(len(ratings)):
        item_item_indices = []
        user_user_indices = []
        item_item_indices.extend(np.argpartition(-item_item_recommendations[user_idx,:], k)[:k])
        user_user_indices.extend(np.argpartition(-user_user_recommendations[user_idx,:], k)[:k])
        ii_actual_ratings.extend([ratings[user_idx][item_idx] for item_idx in item_item_indices])
        uu_actual_ratings.extend([ratings[user_idx][item_idx] for item_idx in user_user_indices])

    predicted_ratings = [1]*(k*1998)
    rmse_for_each_k["item_item_rmse"].append(calc_rmse(np.array(ii_actual_ratings),np.array(predicted_ratings),k*1998))
    rmse_for_each_k["user_user_rmse"].append(calc_rmse(np.array(uu_actual_ratings),np.array(predicted_ratings),k*1998))

rmse_df = pd.DataFrame.from_dict(rmse_for_each_k)
print(f'The method user-user collaborative filtering has RMSE for 5 items: {rmse_for_each_k["user_user_rmse"][4]}')
print(f'The method item-item collaborative filtering has RMSE for 5 items: {rmse_for_each_k["item_item_rmse"][4]}')

ii_plot = sns.lineplot(data=rmse_df,y="item_item_rmse",x="k")
ii_plot.set_ylabel("RMSE(item-item)")
plt.show(block=False)
plt.pause(3000)

uu_plot = sns.lineplot(data=rmse_df,y="user_user_rmse",x="k")
uu_plot.set_ylabel("RMSE(user-user)")
plt.show(block=False)
plt.pause(3000)
