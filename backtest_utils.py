'''
List of utils functions for the launch of backtests - In order to make Jupyter notebooks lighter
'''
import smtplib
import pandas as pd
import numpy as np


gmail_user = 'dorian.lagadec@aequamcapital.com'  
gmail_password = 'Aequaminternship0#'

def get_workable_data(dataframe, length=0, equi_weight=True, dropnan=False):
    data = dataframe[-length:].copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    if dropnan:
        data = data.dropna()
    if equi_weight:
        weight_vector = [1/6 for i in range(6)]
        data['Equi_weighted'] = np.dot(data.iloc[:,0:6], weight_vector)
        
    len_split = int(np.floor(len(data)/2)) 
    train = data[:len_split]
    test = data[len_split:]
    return(data, train, test) 

def ema_and_signal(df, fast_list, slow_list):
    df_temp = df.copy()
    
    for col in df_temp.columns:
        for i in fast_list + slow_list :
            df_temp[col + '_temp_ema_' + str(i)] = df_temp[col].rolling(window=i).mean()
#     df_temp = df_temp.dropna()
        for f in fast_list:
            for l in slow_list:
                df_temp[col + '_signal_' + str(f) + '_' + str(l)] = (df_temp[col + '_temp_ema_' + str(f)] - \
                        df_temp[col + '_temp_ema_' + str(l)] > 0)*1
    df_temp = df_temp.dropna()
    
    for col in df.columns:
        for i in fast_list + slow_list :
            del df_temp[col + '_temp_ema_' + str(i)]
        del df_temp[col]
        
    return df_temp

def send_report(gmail_user, gmail_password):
    sent_from = gmail_user  
    to = ['dorian.lagadec@aequamcapital.com']  
    subject = 'Report'
#     body = ''
    email_text = """Test"""

    try:  
	    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
	    server.ehlo()
	    server.login(gmail_user, gmail_password)
	    server.sendmail(sent_from, to, email_text)
	    server.close()

	    print('Email sent!')
    except:  
        print('Something went wrong...')

def filename(*args):
    return('_'.join([str(i) for i in args]))