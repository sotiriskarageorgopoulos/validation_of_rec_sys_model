import numpy as np
import logging
import scipy.linalg.blas as slb

logging.basicConfig(level = logging.INFO)
rating_matrix = np.genfromtxt('./data/user-shows.txt', delimiter=' ')
logging.info("Rating matrix fetched from txt file...")

#20% of users and items for testing set => 1998 users and 111 items.
rating_matrix[7987:,0:111] = 0

#item-item recommendation matrix
t_matrix = slb.dgemm(alpha=1.0,a=rating_matrix.transpose(),b=rating_matrix)
items_likes = np.sum(rating_matrix ,axis=0)
q_matrix = np.zeros((563,563),float)
np.fill_diagonal(q_matrix,items_likes)
q_sqrt = q_matrix[q_matrix > 0] ** (-1/2)
np.fill_diagonal(q_matrix,q_sqrt)
qt = slb.dgemm(alpha=1.0,a=q_matrix,b=t_matrix)
sim_matrix = slb.dgemm(alpha=1.0,a=qt,b=q_matrix)
item_item_rec_matrix = slb.dgemm(alpha=1.0,a=rating_matrix ,b=sim_matrix)
np.savetxt("item-item-rec-matrix.txt",item_item_rec_matrix,delimiter=' ')
logging.info(f"The item-item recommendation matrix is ready...")

#user-user recommendation matrix
t_matrix = slb.dgemm(alpha=1.0,a=rating_matrix,b=rating_matrix.transpose())
users_likes = np.sum(rating_matrix,axis=1)
p_matrix = np.zeros((9985,9985),float)
np.fill_diagonal(p_matrix,users_likes)
p_sqrt = p_matrix[p_matrix > 0] ** (-1/2) 
np.fill_diagonal(p_matrix,p_sqrt)
pt = slb.dgemm(alpha=1.0,a=p_matrix,b=t_matrix)
sim_matrix = slb.dgemm(alpha=1.0,a=pt,b=p_matrix)
user_user_rec_matrix = slb.dgemm(alpha=1.0,a=sim_matrix,b=rating_matrix)
np.savetxt("user-user-rec-matrix.txt",user_user_rec_matrix,delimiter=' ')
logging.info(f"The user-user recommendation matrix is ready...")