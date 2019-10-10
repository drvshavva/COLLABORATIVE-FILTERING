# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:15:30 2019

@author: Havanur Dervişoğlu

KNN TABANLI FİLM ÖNERİ SİSTEMİ--YZ DERSİ PROJESİ
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import *
from tkinter import messagebox
# utils import
from fuzzywuzzy import fuzz

import matplotlib.pyplot as plt
plt.style.use('ggplot')

##########VERİ SETİNİN YÜKLENMESİ#############################
movies='C:/Users/user/Desktop/Yapay Zeka/ml-latest-small/ml-latest-small/movies.csv'
ratings='C:/Users/user/Desktop/Yapay Zeka/ml-latest-small/ml-latest-small/ratings.csv'

df_movies = pd.read_csv(
    movies,
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(
    ratings,
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})


print("Movies datasetinin özellikleri:")
df_movies.info()
print("\n\n")

print("Ratings datasetinin özellikleri:")
df_ratings.info()
print("\n\n")


num_users = len(df_ratings.userId.unique())
num_items = len(df_ratings.movieId.unique())
print('Bu veri setinde {} tane kullanıcı ve {} film vardır'.format(num_users, num_items))
print("\n\n")

##################ÖN İŞLEME###################################################

#####Her derecelendirmenin kaç tane bulunduğunun plot edilmesi##############
# yeni bir dataframe oluşturuldu, ratingleri sayısıları ile tutan
df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('rating').size(), columns=['count'])

#oluşturulan dataframe de 0 derecelendirme görülmüyor,
#sıfır derecelendirmeyi her kullanıcının her filme oy verdiğini sayıyı
#toplamda yapılan oy sayısını çıkararak elde ederiz ve elde ettiğimiz
#bu sıfır derecelendirme sayısını df eklerix

total_cnt = num_users * num_items
rating_zero_cnt = total_cnt - df_ratings.shape[0]
df_ratings_cnt = df_ratings_cnt_tmp.append(
    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
    verify_integrity=True,
).sort_index()

####bu elde edilen count değerleri karşılaştırmak için birbirlerinden
#uzak değerlere sahip o yüzden log dönüşümü ile bu değerleri birbileri 
#ile karşılaştırabilecek forma getiriyoruz
df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])

ax = df_ratings_cnt[['count']].reset_index().rename(columns={'index': 'rating score'}).plot(
    x='rating score',
    y='count',
    kind='bar',
    figsize=(12, 8),
    title='Her Derecenin Kaç Tane Bulunduğu Sayısı(in Log Scale)',
    logy=True,
    fontsize=12,
)
ax.set_xlabel("movie rating score")
ax.set_ylabel("number of ratings")

########Filmlerin frekanslarının plot edilmesi###########
df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])

ax = df_movies_cnt \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(
        figsize=(12, 8),
        title='Filmlerin Derecelendirilme Frekansı',
        fontsize=12
    )
ax.set_xlabel("movie Id")
ax.set_ylabel("number of ratings")

ax = df_movies_cnt \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(
        figsize=(12, 8),
        title='Filmlerin Derecelendirilme Frekansı (in Log Scale)',
        fontsize=12,
        logy=True
    )
ax.set_xlabel("movie Id")
ax.set_ylabel("number of ratings (log scale)")


##grafiklerde görüldüğü gibi tüm filmlerin 10.000 tanesi 100 kere 
#derecelendirilmiş, 20.000 tanesi 10 kere derecelendirilmiş

####bu yüzden derecelendirilme sayısı az olan filmleri veri setinden
#çıkarırırsak bu hem hafıza sorununu önler hemde az bilinen fimlerin
#silinmesiyle öneri sisteminin kalitesi arttırılmış olur

#20'den az derecelendirilme sayısına sahip filmler silindi
popularity_thres = 20
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
print('Orijinal verinin boyutu: ', df_ratings.shape)
print('Bilinmeyen filmler çıkarıldıktan sonraki verinin boyutu: ', df_ratings_drop_movies.shape)
print("\n\n")


####her kullanıcının verdiği oy sayısına göre aktif ve pasif kullanıcıları
#çıkarmaya çalışıyoruz
df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

ax = df_users_cnt \
    .sort_values('count', ascending=False) \
    .reset_index(drop=True) \
    .plot(
        figsize=(12, 8),
        title='Kullanıcıların Oy verme frekansı(log scale)',
        fontsize=12,
        logy= True
    )
ax.set_xlabel("user Id")
ax.set_ylabel("number of ratings")

#20 oydan az kullanan kullanıcları sliyoruz
ratings_thres = 20
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
print('Orijinal veri boyutu: ', df_ratings.shape)
print('Aktif olamyan kullanıcıları ve az derecelendirilme sayısına sahip olan filmleri cikardiktan sonraki veri boyutu: ', df_ratings_drop_users.shape)
print("\n\n")

##############veriyi sparse matrix haline getiriyoruz#####################################☻
###satırlar kullanıcı ıdler, sütunlar movieid ler ve kesişimleri olan
#veriler ise kullancının o filme verdiği derece değeri

movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)

####her filme index numarası atandı 
movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
}

movie_user_mat_sparse = csr_matrix(movie_user_mat.values)



########################KNN algoritması ile benzer filmlerin bulunması###################
###girilen filmi ve benzerlerini buluyoruz
##fuzzy paketi Levensthein distance hesaplayarak benzerlik oranı bulur
def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
    # eşleşmeleri buluyoruz
    #fuzzy oranı 60'dan fazla olan filmleri eşleşen filmlere ekliyoruz
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # bulunan sonuçlerı sıralıyor
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Aranan film veri setinde yok!!!')
        messagebox.showerror("Error",'Aranan film veri setinde yok!!!')
        return
    if verbose:
        print('Veri tabanında aranan filmle eşleşenler: {0}\n'.format([x[0] for x in match_tuple]))
    ##öneri yapılacak olan fimin indeksini yolluyor
    return match_tuple[0][1]


##cosine benzerliğine göre en benzer k komşusunu listelemek için kullanılan fonksiyon
def knn(movie,k,all_movies):
    distance=[]
    sorted_distance=[]
    for i in range(all_movies.shape[0]):
        #filmler ile girilen film arasındaki cos benzerliğini buluyoruz
        dist=cosine_similarity(movie,all_movies[i,:].toarray())
        distance.append((dist,i))
    #filmlerin benzerliklerinin azalan sırada sıralıyoruz
    sorted_distance=sorted(distance,reverse=True)
    neighbors=[]
    #en yakın k komşusunu alıyoruz
    for x in range(1,k+1):
        neighbors.append((sorted_distance[x][0],sorted_distance[x][1]))
    return distance,neighbors

#####idsi bilinen filmin ismini döndürür
def get_key(val): 
    for key, value in movie_to_idx.items(): 
         if val == value: 
            return key 
        
#####recommender fonksiyonu 
def recommender(data,movie,mapper,n_recommendation):
    ##filmin indexini fuzzy fonksiyonundan alıyoruz
    idx = fuzzy_matching(mapper,movie, verbose=True)    
    print("\n")
    print("Öneri başlatıldı...")
    print('......\n')
    #en yakın k komşusunu alıyoruz
    distance,neighbors=knn(data[idx,:].toarray(),n_recommendation,data)
    for i in range(n_recommendation):
        print('{0}: {1}, benzerlik: {2}'.format(i+1,get_key(neighbors[i][1]), neighbors[i][0]))
    return neighbors
######verinin kaçta kaçı sparse onu bulalım
#movie-user matrisindeki total örnek sayısı
num_entries = movie_user_mat.shape[0] * movie_user_mat.shape[1]
#bu örneklerin kaç tanesi sıfır
num_zeros = (movie_user_mat==0).sum(axis=1).sum()
# tüm örneklerin kaçta kaçı sparse
ratio_zeros = num_zeros / num_entries
print('Verinin {:.2%}\'si sparse\'dır '.format(ratio_zeros))   
#######Arayüz işlemleri#####
def click():
    look_movie=movie.get()  
    Label(root,text="Girilen filme öneriler:",bg="black",fg="white",font="none 12 bold").grid(row=5,column=1,sticky=W)
    neigh=recommender(movie_user_mat_sparse,look_movie,movie_to_idx,10)
    output.delete(0.0,END)
    for i in range(10):
        recommend=str(i+1)+"-"+get_key(neigh[i][1])+"\n\t\tbenzerlik:"+str(neigh[i][0])+"\n"
        output.insert(END,recommend)
    Label(root,text="\n\n\n",bg="black",fg="white",font="none 12 bold").grid(row=18,column=1,sticky=W)
    Button(root,text="Çıkış",width=8,command=close_window).grid(row=18,column=1,sticky=W)
    

def close_window():
    root.destroy()
    exit()
    
root=Tk()
root.title("Yapay Zeka Dersi Projesi")
root.configure(bg="black")
root.geometry("400x650+50+50")

Label(root,text="\t  Film Öneri Sistemi",bg="black",fg="white",font="none 15 bold").grid(row=0,column=1,sticky=W)
foto=PhotoImage(file="C:/Users/user/Desktop/Yapay Zeka/movie-gif.gif")
lb1=Label(root,image=foto,bg="black").grid(row=1,column=1,sticky=W)

Label(root,text="Film ismi giriniz:",bg="black",fg="white",font="none 12 bold").grid(row=2,column=1,sticky=W)
movie=Entry(root,width=20,bg="white")
movie.grid(row=3,column=1,sticky=W)

Label(root,text="\n\n",bg="black",font="none 12 bold").grid(row=4,column=1,sticky=W)

Button(root,text="ÖNER",width=6,command=click).grid(row=4,column=1,sticky=W)
output=Text(root,width=400,height=10,bg="black",fg="white")
output.grid(row=6,column=1,sticky=W)


root.mainloop()  
