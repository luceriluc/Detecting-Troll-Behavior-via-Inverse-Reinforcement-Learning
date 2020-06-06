"""
Utils to:
- Adjust and uniform dataframe to prepare users' trajectories 
- Compute transition probabilities from one state to another when a given action is performed

Luca Luceri, 2020
luca.luceri@supsi.ch
"""

import pandas as pd
import numpy as np
import math
from datetime import timedelta
from datetime import datetime 


def create_user_dataframe(df,user_name,metric,act,kind):
	df_u = df[df[metric]==user_name]
	df_u['activity']=act
	df_u['kind']=kind
	n_tweets = len(df_u)
	#if n_tweets>0:
	df_u['time'] = pd.to_datetime(df_u['created_at'])
	df_u.sort_values(by='time',inplace = True)
	df_u.reset_index(inplace = True)
	del df_u['index']
	return df_u,n_tweets


def adjust_tweet_df(df_u):
	df_u = df_u[['id_str','time','user_followers_count','user_friends_count','user_statuses_count','user_favourites_count','activity','kind']]
	df_u.columns = ['tweet_id','time','followers','friends','statuses','favourites','activity','kind']
	df_u['count']= 0
	df_u['favorites']=0
	df_u['reply']=0
	return df_u

def adjust_retweet_df(df_u):
	df_u = df_u[['time','retweet_id_str','retweet_count','retweet_favorite_count','retweet_reply_count','user_retweeted_followers','user_retweeted_friends','user_retweeted_statuses_count','user_retweeted_favourites_count','activity','kind']]
	df_u.columns = ['time','tweet_id','count','favorites','reply','followers','friends','statuses','favourites','activity','kind']
	return df_u

def adjust_reply_df(df_u):
	df_u = df_u[['time','in_reply_to_status_id_str','activity','kind']]
	df_u.columns = ['time','tweet_id','activity','kind']
	df_u['count']= 0 
	df_u['favorites']= 0
	df_u['reply']=0
	df_u['followers']=0
	df_u['friends']=0
	df_u['statuses']=0 
	df_u['favourites']=0
	df_u=df_u[['time','tweet_id','count','favorites','reply','followers','friends','statuses','favourites','activity','kind']] 
	return df_u

def adjust_mention_df(df_u):
	df_u = df_u[['time','activity','kind']]
	#df_u.columns = ['time','activity','kind']
	df_u['tweet_id']= 0   
	df_u['count']= 0 
	df_u['favorites']= 0
	df_u['reply']=0
	df_u['followers']=0
	df_u['friends']=0
	df_u['statuses']=0 
	df_u['favourites']=0
	df_u=df_u[['time','tweet_id','count','favorites','reply','followers','friends','statuses','favourites','activity','kind']] 
	return df_u

def merge_df(df_b,df_RT_mnt):
	df_total = df_b.append(df_RT_mnt)
	df_total = df_total.sort_values(by='time') 
	df_total.reset_index(inplace = True)
	del df_total['index']
	df_total=df_total.rename(columns = {'favorites':'tweet_likes'})
	df_total=df_total.rename(columns = {'count':'retweet_count'})
	return df_total

def compute_tp(state_sequence,n_states,n_actions):
	tp = np.zeros([n_states,n_actions,n_states])

	for pair in np.arange(len(state_sequence)-1):
	    s=state_sequence[pair][0]
	    a=state_sequence[pair][1]
	    ns=state_sequence[pair+1][0]
	    tp[s,a,ns]+=1

	for c in np.arange(len(tp)):
	    A= tp[c]/tp[c].sum(axis=1)[:,None]
	    A[np.isnan(A)] = 0
	    tp[c]= A

	return tp





