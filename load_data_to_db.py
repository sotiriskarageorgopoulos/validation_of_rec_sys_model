#author Sotirios Karageorgopoulos
import logging
import json
from random import choice
import numpy as np
from dotenv import load_dotenv
from py2neo import Graph,Node,Relationship
import os

load_dotenv()
logging.basicConfig(level = logging.INFO)

shows = []
with open('./data/shows.txt','r') as file:
    shows_txt = file.read()
    shows = list(filter(lambda x: x != '',shows_txt.split("\n")))
logging.info("shows fetched from txt file...")

rating_matrix = np.genfromtxt('./data/user-shows.txt', delimiter=' ')
logging.info("Rating matrix fetched from txt file...")

item_item_rec_matrix = np.genfromtxt('./data/item-item-rec-matrix.txt', delimiter=' ')
logging.info("Item-Item recommendation matrix fetched from txt file...")

user_user_rec_matrix = np.genfromtxt('./data/user-user-rec-matrix.txt', delimiter=' ')
logging.info("User-User recommendation matrix fetched from txt file...")

users_names = []
with open('./data/names.txt','r') as file:
    users_names_txt = file.read()
    users_names = list(filter(lambda x: x != '',users_names_txt.split("\n")))
logging.info("names of users fetched from txt file...")

user_surnames = []
with open('./data/surnames.txt','r') as file:
    user_surnames_txt = file.read()
    user_surnames = list(filter(lambda x: x != '',user_surnames_txt.split("\n")))
logging.info("surnames of users fetched from txt file...")  

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
g = Graph("http://localhost:7474",user=db_user,password=db_password)

shows_nodes = []
with open('./data/shows_images.json','r') as file:
    json_lst = json.load(file)
    shows_id = 0
    for s in shows:
        img = list(filter(lambda show: "\""+show["title"]+"\"" == s,json_lst))
        if len(img) > 0:
            shows_nodes.append(Node("Show", show_id=shows_id,title=s,image=img[0]["image"]))
        else:
            shows_nodes.append(Node("Show", show_id=shows_id,title=s,image=None))
        shows_id += 1
        
logging.info(f"Shows with images were created...")


keyboard_chars = "!\"/0123456789@ABCDEFGHIJKLMNOPQRSTUVWXYZ_`abcdefghijklmnopqrstuvwxyz"
user_id = 0

for user_ratings in rating_matrix:
    tx = g.begin() 
    name = choice(users_names)
    surname = choice(user_surnames)
    email = surname.lower()+"@domain.com"
    password = ""
            
    for i in range(10):
        password += choice(keyboard_chars)
            
    user = Node("User",user_id=user_id,name=name,surname=surname,email=email,password=password)
    
    for j in range(len(user_ratings)):
        rated = Relationship(user,"RATED",shows_nodes[j],rating=user_ratings[j])
        reccomended = Relationship(shows_nodes[j],"RECOMMENDED",user,item_item_rec=item_item_rec_matrix[user_id,j],
                    user_user_rec=user_user_rec_matrix[user_id][j])
        tx.create(rated)
        tx.create(reccomended)
       
    tx.commit()
    logging.info(f"The user {user_id} was added...")
    user_id += 1