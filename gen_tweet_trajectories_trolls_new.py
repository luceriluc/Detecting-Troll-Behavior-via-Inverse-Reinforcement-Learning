"""
Find the trajectories based on trolls' (users') history
It returns user's trajectories, sequence of visited states, number of visited states,number of actions performed, the feature matrix, and some dictionaries

Line 89/90 and 92/93 should be commented/uncommented based on the used Python version (2 vs. 3)

Luca Luceri, 2020
luca.luceri@supsi.ch
"""


import pandas as pd
import numpy as np
import math
from datetime import timedelta
from datetime import datetime 


def tweet_traj_next_reduced(df_total):
    # initialize dictionaries and parameters
    s_dict={0: [0, 0, 0, 0, 0],1: [0, 0, 0, 0, 1],2: [0, 0, 0, 1, 0], 3: [0, 0, 1, 0, 0], 4: [0, 1, 0, 0, 0], 5: [0,1,0,0,1],6:[0, 1, 0, 1, 0], 7: [0, 1, 1, 0, 0], 8: [1, 0, 0, 0, 0], 9: [1, 0, 0, 0, 1], 10: [1, 0, 0, 1, 0],11: [1, 0, 1, 0, 0]} #feature matrix preparation
    a_dict = a_dict={0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3: [1, 0, 0]} # actions dictionary
    t_dict = {} # tweet dictionary
    feature_matrix=[] # feature matrix preparation
    for el in s_dict:
        array=s_dict[el]
        int_array = [ int(s) for s in array ]
        feature_matrix.append(int_array)
    feature_matrix = np.asarray(feature_matrix)   
    c_dict={}
    for i in np.arange(len(s_dict)):
        c_dict[i]=0

    n_actions = len(a_dict)
    n_states = len(s_dict)
    
    # initialize state label 
    tw_count_p = df_total.loc[0].statuses    
    
    # tweet dictionary initialization
    tw_1=df_total.loc[0].tweet_id
    if tw_1>0:
        t_dict[tw_1]={}
        t_dict[tw_1]['rt']=df_total.loc[0].retweet_count
        t_dict[tw_1]['rp']=df_total.loc[0].reply

    # initialize action features (signal = 0 --> state, signal = 1 --> action)
    if (df_total.loc[0].activity ==1):    # action: original tweet    
        action_features_0 = np.asarray([1,0,0])
        state_ = [0,0]
        signal=1
    elif (df_total.loc[0].activity ==2): # action: retweet
        action_features_0 = np.asarray([0,1,0])
        state_ = [0,0]
        signal=1
    elif ((df_total.loc[0].activity ==3) or (df_total.loc[0].activity ==4)): # action: reply or mention
        action_features_0 = np.asarray([0,0,1])
        state_ = [0,0]
        signal=1
    elif (df_total.loc[0].activity ==5): # state: got a retweet
        action_features_0 = np.asarray([0,0,0])
        state_ = [1,0]
        signal=0
    elif ((df_total.loc[0].activity ==6) or (df_total.loc[0].activity ==7)): # state: got a reply or mention
        action_features_0 = np.asarray([0,0,0])
        state_ = [0,1]
        signal=0
    
    # trajectory length  
    default_length = 100 # set as a default value, it can be edited based on users' activity 

    if len(df_total)>default_length:
        trajectory_length = default_length
    else:
        trajectory_length = len(df_total)-1
        
    trajectories =[]
    state_sequence = []
    period_traj = []
    
    for t in np.arange(len(df_total))[1:]:   
        tw_count_t=df_total.loc[t].statuses
        if signal==0: # if at time t-1 there was a passive event check if there are new tweets
            tw_delta = tw_count_t-tw_count_p
            if tw_delta>0:            
                action_features_0[0]+=tw_delta
            
        action_t = np.where(action_features_0 > 0, 1, 0)
        action = list(a_dict.keys())[list(a_dict.values()).index(list(action_t))] #python3
        #action = a_dict.keys()[a_dict.values().index(list(action_t))] # python2
        state_t = np.concatenate((state_, action_t), axis=0)
        state = list(s_dict.keys())[list(s_dict.values()).index(list(state_t))] #python3
        # state = s_dict.keys()[s_dict.values().index(list(state_t))]  # python2
        c_dict[state]+=1
        period_traj.append([state,action])
        state_sequence.append([state,action])
        tw_count_p = tw_count_t        
        
        # state after the action            
        rt_count=df_total.loc[t].retweet_count 
        rp_count=df_total.loc[t].reply
        if np.isnan(rp_count):
            rp_count=0    
        tw_t=int(df_total.loc[t].tweet_id)
        if tw_t>0:
            if tw_t in t_dict:
                # take the previous info
                rt_count_p = t_dict[tw_t]['rt']
                rp_count_p = t_dict[tw_t]['rp']
                rt_count_t=rt_count-rt_count_p
                rp_count_t=rp_count-rp_count_p
            else:
                # add tweet to the dictionary
                t_dict[tw_t]={}
                t_dict[tw_t]['rt']=rt_count
                t_dict[tw_t]['rp']=rp_count
                rt_count_t=0
                rp_count_t=0
        if df_total.loc[t].kind=="p":
            if df_total.loc[t].activity==5: # state: got a retweet
                rt_count_t=1
                state_ = np.asarray([rt_count_t,0])
            elif (df_total.loc[t].activity==6) or (df_total.loc[t].activity==7): # state: got a reply or mention
                rp_count_t=1
                state_ = np.asarray([0,rp_count_t])
        else:
            state_ = np.asarray([rt_count_t,rp_count_t])
        state_ = np.where(state_ > 0, 1, 0)
        
        if (df_total.loc[t].activity ==1): # action: original tweet 
            action_features_0 = np.asarray([1,0,0])
            signal=1
        elif (df_total.loc[0].activity ==2): # action: retweet 
            action_features_0 = np.asarray([0,1,0])
            signal=1
        elif ((df_total.loc[0].activity ==3) or (df_total.loc[0].activity ==4)): # action: reply or mention 
            action_features_0 = np.asarray([0,0,1])
            signal=1
        else:
            action_features_0 = np.asarray([0,0,0]) # action: nothing
            signal=0


        if t%trajectory_length==0:
            trajectories.append(period_traj) # trajectory story
            period_traj = []

    trajectories = np.asarray(trajectories)
    return trajectories,state_sequence,n_states,n_actions,feature_matrix,t_dict,c_dict

