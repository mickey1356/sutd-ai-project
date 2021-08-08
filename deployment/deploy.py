import numpy as np
import praw
import pandas as pd
import datetime as dt
import re
from decouple import config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from apscheduler.schedulers.background import BackgroundScheduler

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import pyrebase
import time
from datetime import date

import yfinance as yf
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def get_new_1000(reddit):
    """Gets the 1000 most recent posts to r/wallstreetbets

    Args:
        reddit (Reddit): A Reddit instance as defined by `praw`

    Returns:
        tuple of lists of dictionaries: In order, the posts that contain mentions of BB, AMC, NOK, GME, and any of them.
        The dictionaries are defined exactly as specified in `scrape-reddit.py` in the development folder.
    """    
    posts = reddit.subreddit('wallstreetbets').new(limit=1000)

    bb_f = ['bb', 'blackberry']
    amc_f = ['amc']
    nok_f = ['nok', 'nokia']
    gme_f = ['gme', 'gamestop']

    bbs = []
    amcs = []
    noks = []
    gmes = []
    alls = []

    for p in posts:
        a = p.author
        c = p.created_utc
        i = p.id
        t = p.title
        po = p.selftext
        n = p.num_comments
        s = p.score
        r = p.upvote_ratio
        if po != '':
            if any(f in re.split('\W+', po.lower()) for f in bb_f):
                bbs.append((a,c,i,t,po,n,s,r))
            if any(f in re.split('\W+', po.lower()) for f in amc_f):
                amcs.append((a,c,i,t,po,n,s,r))
            if any(f in re.split('\W+', po.lower()) for f in nok_f):
                noks.append((a,c,i,t,po,n,s,r))
            if any(f in re.split('\W+', po.lower()) for f in gme_f):
                gmes.append((a,c,i,t,po,n,s,r))
            if any(f in re.split('\W+', po.lower()) for f in bb_f + amc_f + nok_f + gme_f):
                alls.append((a,c,i,t,po,n,s,r))

    return bbs, amcs, noks, gmes, alls

def get_data(reddit):
    """Runs every hour (at the 20 minute mark) in order to scrape the data from Reddit.
    Will also use `sentiment.pkl` to assign a sentiment score for each post.
    Adds the data into a globally-defined dataframe.

    Args:
        reddit (Reddit): A Reddit instance as defined by `praw`
    """    
    global df
    global df_bb
    global df_amc
    global df_nok
    global df_gme

    print('go')
    bbs, amcs, noks, gmes, alls = get_new_1000(reddit)
    
    to_add = pd.DataFrame(alls, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    to_add['c'] = pd.to_datetime(to_add['c'], unit='s')
    
    bb_to_add = pd.DataFrame(bbs, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    bb_to_add['c'] = pd.to_datetime(bb_to_add['c'], unit='s')
    
    amc_to_add = pd.DataFrame(amcs, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    amc_to_add['c'] = pd.to_datetime(amc_to_add['c'], unit='s')
    
    nok_to_add = pd.DataFrame(noks, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    nok_to_add['c'] = pd.to_datetime(nok_to_add['c'], unit='s')
    
    gme_to_add = pd.DataFrame(gmes, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    gme_to_add['c'] = pd.to_datetime(gme_to_add['c'], unit='s')
    
    #to resolve empty df
    data = [['author', dt.datetime.now(), 'xxx', 'title', 'post', 0, 0, 0]]
    if (len(bb_to_add)==0):
      bb_to_add = pd.DataFrame(data, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    if (len(amc_to_add)==0):
      amc_to_add = pd.DataFrame(data, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    if (len(nok_to_add)==0):
      nok_to_add = pd.DataFrame(data, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])
    if (len(gme_to_add)==0):
      gme_to_add = pd.DataFrame(data, columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r'])

    '''
    Add the sentiment analysis, pre-train
     analyser = SentimentIntensityAnalyzer()
     def sentiment_analyzer_scores(row):
         score = analyser.polarity_scores(row['p'])
         score = float(str(score['compound']))
         if score != 0:
             new_score = score * 2
             if new_score > 0:
                 new_score += 0.5
             else:
                 new_score -= 0.5
         else:
             new_score = 0
         return int(new_score)
    '''
        
    #home-made, self train
    with open("sentiment.pkl", 'rb') as f:
        vectorizer, sentimental_model = pickle.load(f)
    def sentiment_analyzer_scores2(row):
        test_post = vectorizer.transform([row['p']])
        prediction = sentimental_model.predict(test_post)
        return prediction[0]
    
    to_add['sentiment'] = to_add.apply(sentiment_analyzer_scores2, axis=1)
    bb_to_add['sentiment'] = bb_to_add.apply(sentiment_analyzer_scores2, axis=1)
    amc_to_add['sentiment'] = amc_to_add.apply(sentiment_analyzer_scores2, axis=1)
    nok_to_add['sentiment'] = nok_to_add.apply(sentiment_analyzer_scores2, axis=1)
    gme_to_add['sentiment'] = gme_to_add.apply(sentiment_analyzer_scores2, axis=1)

    # drop any na values
    to_add.dropna(inplace=True)
    bb_to_add.dropna(inplace=True)
    amc_to_add.dropna(inplace=True)
    nok_to_add.dropna(inplace=True)
    gme_to_add.dropna(inplace=True)

    # append the new values to a global dataframe
    df = df.append(to_add)
    df = df.drop_duplicates(subset=['i'])
    df_bb = df_bb.append(bb_to_add)
    df_bb = df_bb.drop_duplicates(subset=['i'])
    df_amc = df_amc.append(amc_to_add)
    df_amc = df_amc.drop_duplicates(subset=['i'])
    df_nok = df_nok.append(nok_to_add)
    df_nok = df_nok.drop_duplicates(subset=['i'])
    df_gme = df_gme.append(gme_to_add)
    df_gme = df_gme.drop_duplicates(subset=['i'])

    # it is highly unlikely that we have two posts posted at the exact same time
    # we do this so we can filter by date
    # df = df.set_index(['c'])
    # df_bb = df_bb.set_index(['c'])
    # df_amc = df_amc.set_index(['c'])
    # df_nok = df_nok.set_index(['c'])
    # df_gme = df_gme.set_index(['c'])

    #print(df)
    print('Overall reddit data: ', len(df))
    print('Reddit data (BB): ', len(df_bb))
    print('Reddit data (AMC): ', len(df_amc))
    print('Reddit data (NOK): ', len(df_nok))
    print('Reddit data (GME): ', len(df_gme))

    print('ok')

def get_top_sentiments2(df):
    """Gets the highest-rated sentiments in a DataFrame.
    Used in order to upload to Firebase for visualisation.

    Args:
        df (pd.DataFrame): A dataframe object containing the post data as well as its sentiment label generated by the sentiment model.

    Returns:
        Tuple of dictionaries: A tuple of dictionaries containing the posts that result in the most extreme sentiments.
    """    

    df_sen_sorted = df.sort_values(by='sentiment')
    p_list = df_sen_sorted['p'].tolist()
    s_list = df_sen_sorted['sentiment'].to_list()
    
    try:
        if (s_list[0] < 0):
            neg_sen1 = {'post': p_list[0] , 'sen_score': s_list[0]}
        else:
            neg_sen1 = {'post': 'Nil' , 'sen_score': 0}
    except:
        neg_sen1 = {'post': 'Nil' , 'sen_score': 0}
    try:
        if (s_list[1] < 0):
            neg_sen2 = {'post': p_list[1] , 'sen_score': s_list[1]}
        else:
            neg_sen2 = {'post': 'Nil' , 'sen_score': 0}
    except:
        neg_sen2 = {'post': 'Nil' , 'sen_score': 0}
    try:
        if (s_list[2] < 0):
            neg_sen3 = {'post': p_list[2] , 'sen_score': s_list[2]}
        else:
            neg_sen3 = {'post': 'Nil' , 'sen_score': 0}
    except:
        neg_sen3 = {'post': 'Nil' , 'sen_score': 0}
    #-----------------------------------------------------------------------------------------
    try:
        if (s_list[len(p_list)-1] > 0):
            pos_sen1 = {'post': p_list[len(p_list)-1] , 'sen_score': s_list[len(p_list)-1]}
        else:
            pos_sen1 = {'post': 'Nil' , 'sen_score': 0}
    except:
        pos_sen1 = {'post': 'Nil' , 'sen_score': 0}
    try:
        if (s_list[len(p_list)-2] > 0):
            pos_sen2 = {'post': p_list[len(p_list)-2] , 'sen_score': s_list[len(p_list)-2]}
        else:
            pos_sen2 = {'post': 'Nil' , 'sen_score': 0}
    except:
        pos_sen2 = {'post': 'Nil' , 'sen_score': 0}
    try:
        if (s_list[len(p_list)-3] > 0):
            pos_sen3 = {'post': p_list[len(p_list)-3] , 'sen_score': s_list[len(p_list)-3]}
        else:
            pos_sen3 = {'post': 'Nil' , 'sen_score': 0}
    except:
        pos_sen3 = {'post': 'Nil' , 'sen_score': 0}
        
    
    return neg_sen1, neg_sen2, neg_sen3, pos_sen1, pos_sen2, pos_sen3

def get_sentiment_count(sen_list):
    """Gets the number of postive, negative, and zero sentiments.

    Args:
        sen_list (list): A list containing just the sentiment labels assigned to each post.

    Returns:
        dict: A dictionary containing the number of sentiments which are postive, negative, and zero.
    """
    positive_count = list(filter(lambda score: score > 0, sen_list))
    negative_count = list(filter(lambda score: score < 0, sen_list))
    neutral_count = list(filter(lambda score: score == 0, sen_list))

    return {'positive_count':len(positive_count), 'negative_count':len(negative_count), 'neutral_count':len(neutral_count)}
  
  
def get_model():
    """Defines and loads the neural network model used to predict price movements.

    Returns:
        nn.Module: A PyTorch model loaded with the weights in `bang_model`.
    """    
    class FeedforwardNeuralNetModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(FeedforwardNeuralNetModel, self).__init__()
            
            hidden_dim1 = 4
            hidden_dim2 = 2
            
            # Linear function
            self.fc1 = nn.Linear(input_dim, hidden_dim1) 

            # Linear function
            self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) 
            
            # Linear function
            self.fc3 = nn.Linear(hidden_dim2, output_dim) 
        

        def forward(self, x):
            
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
        
            return out
    
    bang_model = FeedforwardNeuralNetModel(4,1)
    bang_model.load_state_dict(torch.load('bang_model'))
    
    return bang_model
    

def get_model_output(X):
    """Given some input data, get the model's output for that data.
    Will instantiate and load the model from the weights, and pass the input data through the loaded model.

    Args:
        X (np.array): A numpy array containing the input features to be passed into the model.

    Returns:
        torch.Tensor: A PyTorch tensor variable containing the output of the model.
    """    
    new_input = torch.tensor(X)
    # load model
    bang_model = get_model()
    output = bang_model(new_input.float())
    return output
    
def get_price_change(data):
    """Builds an input Tensor using a given dataframe object, in order to get the predicted price change as produced by the model.

    Args:
        data (pd.DataFrame): A dataframe object containing the post data as well as its sentiment label generated by the sentiment model.

    Returns:
        tuple of torch.Tensors: A tuple containing the input tensor, as well the output tensor produced by the model.
    """    
    # i use the average, but this can be changed to sum or whatever other function
    n = data['n'].mean()
    s = data['s'].mean()
    r = data['r'].mean()
    sents = data['sentiment'].mean()
    # pack it into a numpy array
    X = np.nan_to_num(np.array([n, s, r, sents]))
    output = get_model_output(X)
    output = output.item()
    #resolve nan
    if (output!=output):
        output = 0
    X = torch.tensor(X)
    X = X.float()
    return X, output
    
def run_model():
    """Runs every weekday at 1.30pm GMT (market open time).
    Will use the globally defined dataframes and filter them by the timestamps (we only use the posts made during when the market was closed).
    Uploads the results to firebase.
    """    
    print('go2')
    global df
    global df_bb
    global df_amc
    global df_nok
    global df_gme
    
    global bb_model_data
    global amc_model_data
    global nok_model_data
    global gme_model_data

    # find today
    now = dt.datetime.utcnow()
    today = now.weekday()

    # find the date of the previous weekday (for date filtering)
    # if today is monday (0) or sunday (6) the previous weekday is -3 (for friday) and -2 (for friday)
    prev_delta = [-3, -1, -1, -1, -1, -1, -2][today]
    last_weekdate = now.date() + dt.timedelta(days=prev_delta)

    # get the actual start and end times of the period we are looking at
    last_time = dt.datetime(year=last_weekdate.year, month=last_weekdate.month, day=last_weekdate.day) + dt.timedelta(hours=20)
    now_time = dt.datetime(year=now.date().year, month=now.date().month, day=now.date().day) + dt.timedelta(hours=13, minutes=30)

    print('dates: ', last_time, now_time)

    # it is highly unlikely that we have two posts posted at the exact same time
    # we do this so we can filter by date
    df_dt = df.set_index(['c'])
    df_bb_dt = df_bb.set_index(['c'])
    df_amc_dt = df_amc.set_index(['c'])
    df_nok_dt = df_nok.set_index(['c'])
    df_gme_dt = df_gme.set_index(['c'])

    # sort index
    df_dt.sort_index(inplace=True)
    df_bb_dt.sort_index(inplace=True)
    df_amc_dt.sort_index(inplace=True)
    df_nok_dt.sort_index(inplace=True)
    df_gme_dt.sort_index(inplace=True)

    # filter posts by their date
    date_filtered = df_dt.loc[last_time:now_time]
    bb_date_filtered = df_bb_dt.loc[last_time:now_time]
    amc_date_filtered = df_amc_dt.loc[last_time:now_time]
    nok_date_filtered = df_nok_dt.loc[last_time:now_time]
    gme_date_filtered = df_gme_dt.loc[last_time:now_time]

    print('amt of data: ', len(date_filtered), len(bb_date_filtered), len(amc_date_filtered), len(nok_date_filtered), len(gme_date_filtered))

    bb_X, bb_price_change = get_price_change(bb_date_filtered)
    amc_X, amc_price_change = get_price_change(amc_date_filtered)
    nok_X, nok_price_change = get_price_change(nok_date_filtered)
    gme_X, gme_price_change = get_price_change(gme_date_filtered)
    
    bb_model_data['input'] = bb_X
    amc_model_data['input'] = amc_X
    nok_model_data['input'] = nok_X
    gme_model_data['input'] = gme_X
        
    print('price changes: ', bb_price_change, amc_price_change, nok_price_change, gme_price_change)    
    
    #Upload to firebase
    firebase_config = { "apiKey": config('FIREBASE_API_KEY'),
               "authDomain": config('FIREBASE_AUTH_DOMAIN'),
               "databaseURL": config('FIREBASE_DB_URL'),
               "storageBucket": config('FIREBASE_STORAGE_BUCKET') }

    firebase = pyrebase.initialize_app(firebase_config)
    db = firebase.database()
    
    bb_sentiment_count_dict = get_sentiment_count(bb_date_filtered['sentiment'].tolist())
    amc_sentiment_count_dict = get_sentiment_count(amc_date_filtered['sentiment'].tolist())
    nok_sentiment_count_dict = get_sentiment_count(nok_date_filtered['sentiment'].tolist())
    gme_sentiment_count_dict = get_sentiment_count(gme_date_filtered['sentiment'].tolist())
    neg_sen1, neg_sen2, neg_sen3, pos_sen1, pos_sen2, pos_sen3 = get_top_sentiments2(date_filtered)
    
    today = date.today()
    #print("Today's date:", today)
    
    try:
        
        update_indicator = db.child(str(today)).child("BB").set({
            'pc': bb_price_change,
            'positive_count': bb_sentiment_count_dict['positive_count'],
            'negative_count': bb_sentiment_count_dict['negative_count'],
            'neutral_count': bb_sentiment_count_dict['neutral_count']
        })
        
        update_indicator = db.child(str(today)).child("AMC").set({
            'pc': amc_price_change,
            'positive_count': amc_sentiment_count_dict['positive_count'],
            'negative_count': amc_sentiment_count_dict['negative_count'],
            'neutral_count': amc_sentiment_count_dict['neutral_count']
        })
        
        update_indicator = db.child(str(today)).child("NOK").set({
            'pc': nok_price_change,
            'positive_count': nok_sentiment_count_dict['positive_count'],
            'negative_count': nok_sentiment_count_dict['negative_count'],
            'neutral_count': nok_sentiment_count_dict['neutral_count']
        })
        
        update_indicator = db.child(str(today)).child("GME").set({
            'pc': gme_price_change,
            'positive_count': gme_sentiment_count_dict['positive_count'],
            'negative_count': gme_sentiment_count_dict['negative_count'],
            'neutral_count': gme_sentiment_count_dict['neutral_count']
        })
        
        update_indicator = db.child(str(today)).child("sentiments").set({

            'neg_sen1': neg_sen1['post'],
            'neg_sen1_score': neg_sen1['sen_score'],
            'neg_sen2': neg_sen2['post'],
            'neg_sen2_score': neg_sen2['sen_score'],
            'neg_sen3': neg_sen3['post'],
            'neg_sen3_score': neg_sen3['sen_score'],

            'pos_sen1': pos_sen1['post'],
            'pos_sen1_score': pos_sen1['sen_score'],
            'pos_sen2': pos_sen2['post'],
            'pos_sen2_score': pos_sen2['sen_score'],
            'pos_sen3': pos_sen3['post'],
            'pos_sen3_score': pos_sen3['sen_score']
        })
    except:
        print('Firebase upload fail')
    
    # we can clear df because the next period will not use any of the data being used here
    df = pd.DataFrame()
    df_bb = pd.DataFrame()
    df_amc = pd.DataFrame()
    df_nok = pd.DataFrame()
    df_gme = pd.DataFrame()

    print('ok2')

def get_actual_price_change(stock):
    """Gets the actual price change for a given ticker after the market has closed.

    Args:
        stock (string): The symbol of the ticker to get the day's price change for.

    Returns:
        float: The percentage price change of a given ticker in that day's market.
    """    
    today = date.today()
    df = yf.download(stock, 
                        start=str(today), 
                        end=str(today), 
                        progress=False)
    price_change = ((df['Close'] - df['Open']) / df['Open'])[0]
    
    return price_change
    
def update_model():
    """Runs every weekday at 10pm GMT (2 hours after the market has closed).
    The reason why we wait 2 hours is to allow Yahoo! finance to update their price data.
    Retrains and updates the weights of the model according to the current day's data.
    Also uploads the actual price changes to Firebase as well.
    """
    global bb_model_data
    global amc_model_data
    global nok_model_data
    global gme_model_data

    print('Updating Model')
    
    bang_model = get_model()
    
    bb_actual_price_change = get_actual_price_change('BB')
    amc_actual_price_change = get_actual_price_change('AMC')
    nok_actual_price_change = get_actual_price_change('NOK')
    gme_actual_price_change = get_actual_price_change('GME')
    
    data_update = [[bb_model_data['input'], bb_actual_price_change], 
                  [amc_model_data['input'], amc_actual_price_change],   
                  [nok_model_data['input'], nok_actual_price_change],   
                  [gme_model_data['input'], gme_actual_price_change]]
    
    try:
        criterion = nn.MSELoss()
        learning_rate = 0.01
        optimizer = torch.optim.Adam(bang_model.parameters(), lr=learning_rate)  
        for data in data_update:
            
            #Clear gradients 
            optimizer.zero_grad()
            #print(data)
            #Forward pass to get output
            outputs = bang_model(data[0])

            #Calculate Loss with cross entropy loss function
            loss = criterion(outputs.float(), torch.tensor([data[1]]).float())

            #Getting gradients and updating parameters with backpropagation
            loss.backward()
            optimizer.step()
            
        #save updated model
        torch.save(bang_model.state_dict(), 'bang_model')
        print('Model update done')
    except:
        print('Model update failed')
    
    #clear the data for updating
    bb_model_data = {}
    amc_model_data = {}
    nok_model_data = {}
    gme_model_data = {}
    
    firebase_config = { "apiKey": config('FIREBASE_API_KEY'),
               "authDomain": config('FIREBASE_AUTH_DOMAIN'),
               "databaseURL": config('FIREBASE_DB_URL'),
               "storageBucket": config('FIREBASE_STORAGE_BUCKET') }

    firebase = pyrebase.initialize_app(firebase_config)
    db = firebase.database()
    
    #send accurate data to the database
    today = date.today()
    
    try:
        update_indicator = db.child(str(today)).child("BB").update({
            'pc': bb_actual_price_change
        })
        update_indicator = db.child(str(today)).child("AMC").update({
            'pc': amc_actual_price_change
        })
        update_indicator = db.child(str(today)).child("NOK").update({
            'pc': nok_actual_price_change
        })
        update_indicator = db.child(str(today)).child("GME").update({
            'pc': gme_actual_price_change
        })
    except:
        print('Firebase upload fail')

    print('updating done')


def main():
    """Initialises a Reddit instance.
    Also schedules three tasks as defined above.
    The first runs at the 20 minute mark every hour every day in order to scrape Reddit data.
    The second runs at 1.30pm GMT every weekday in order to run the model and obtain that day's predictions.
    The third runs at 10pm GMT every weekday in order to update the model with the day's actual price changes.
    """    
    reddit = praw.Reddit(
        client_id=config('REDDIT_CLIENT_ID'),
        client_secret=config('REDDIT_CLIENT_SECRET'),
        user_agent=config('REDDIT_CLIENT_NAME')
    )

    global df
    df = pd.DataFrame(columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r', 'sentiment'])
    global df_bb
    df_bb = pd.DataFrame(columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r', 'sentiment'])
    global df_amc
    df_amc = pd.DataFrame(columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r', 'sentiment'])
    global df_nok
    df_nok = pd.DataFrame(columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r', 'sentiment'])
    global df_gme
    df_gme = pd.DataFrame(columns= ['a', 'c', 'i', 't', 'p', 'n', 's', 'r', 'sentiment'])
    
    global bb_model_data
    bb_model_data = {}
    global amc_model_data
    amc_model_data = {}
    global nok_model_data
    nok_model_data = {}
    global gme_model_data
    gme_model_data = {}


    # market opens from 1.30pm gmt to 8pm gmt
    # everytime the model runs (we will run it at 1.30pm gmt on weekdays), it takes as input the data from 8pm gmt of the previous day to 1.30pm of today
    # if today is monday, then it takes as input the data from 8pm gmt of last friday, to 1.30pm of monday

    # this runs in gmt
    sched = BackgroundScheduler(timezone='UTC')

    # we schedule two jobs
    # the first runs every hour to grab data
    # we will run this every hour at the 20 minute mark (so we can get one last set of data before the model runs)
    # we give a 10 minute leeway in case of API slowdown or whatnot
    # sched.add_job(get_data, trigger='interval', seconds=20, args=[reddit]) # to the test the function
    sched.add_job(get_data, trigger='cron', minute=20, args=[reddit])

    # the second job will run at 1.30pm gmt every weekday
    # sched.add_job(run_model, trigger='interval', seconds=30) # this is to test the function
    sched.add_job(run_model, trigger='cron', day_of_week='mon,tue,wed,thu,fri', hour=13, minute=30)
    # sched.add_job(run_model, trigger='cron', day_of_week='mon,tue,wed,thu,fri', hour=13, minute=55)

    # the third job will run at 10pm gmt
    # sched.add_job(run_model, trigger='interval', seconds=30) # this is to test the function
    sched.add_job(update_model, trigger='cron', day_of_week='mon,tue,wed,thu,fri', hour=22)


    print('started')

    sched.start()

    try:
        while True:
            continue
    except KeyboardInterrupt:
        pass

    print(df)

if __name__ == '__main__':
    main()
