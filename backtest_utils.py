'''
List of utils functions for the launch of backtests - In order to make Jupyter notebooks lighter
'''
import smtplib
import pandas as pd
import numpy as np

#Credentials for automatic mail reporting
gmail_user = 'dorian.lagadec@aequamcapital.com'  
gmail_password = 'Aequaminternship0#'

def rescale(df, start_int = 0, base = 100):
    '''
    Args :
        df (pandas.DataFrame): the DataFrame to rescale
        start_int (int): the row that you want to set as base (as benchmark)
        base (float): the base you want to have for evaluation purposes

    Returns :
        a pandas.DataFrame rescaled as needed
    ''' 
    return(df/np.array(df.iloc[start_int,:])*base)

def propagate_index(from_df, to_df):
    '''
    Args :
        from_df (pandas.DataFrame): the DataFrame that is used for common index
        to_df (pandas.DataFrame): the DataFrame you want to truncate to fit the other DataFrame's schema

    Returns :
        a pandas.DataFrame matching the truncated to_df
    '''
    return(to_df.loc[from_df.index,:])

def get_workable_data(dataframe, length=0, equi_weight=True, dropnan=False):
    '''
    MAYBE OBSOLETE : useful if you use pandas.read_csv, but not if you use pandas.read_pickle
    Is used to get a proper pandas.DataFrame with proper formats for index, columns, values, ...

    Args :
        dataframe (pandas.DataFrame): the DataFrame to work with
        length (int): >0 only if you want to take the last 'length' rows
        equi_weight (boolean): True if you want to add an equi_weighted column
        dropnan (boolean): True if you want to drop NaN values at the end of the process

    Returns :
        data (pandas.DataFrame): the original dataframe properly formatted and truncated (and enriched)
        train (pandas.DataFrame): the first half of data
        test (pandas.DataFrame): the second half of data
    ''' 
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
    '''
    Args :
        df (pandas.DataFrame): the DataFrame that you want to perform exponential moving average on
        fast_list (list of int): list with small integers for fast moving averages
        slow_list (list of int): list with big integers for slow moving averages

    Returns :
        a pandas.DataFrame with ema differences (pointwise between small and big EMA windows)
    '''
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

def send_report(gmail_user, gmail_password, to = ['dorian.lagadec@aequamcapital.com'] ):
    '''
    For reporting purposes (can be useful when you leave the office and want to check if
    the code is still up and running)
    You can enrich email_text to make a more informative email

    Args :
        gmail_user (str): the gmail address you want to send the reports from
        gmail_password (str): the gmail password associated to the gmail user account
        to (list of str): the mail adresses you want to send the report to

    Returns :
        NOTHING (only send an email)
    '''

    sent_from = gmail_user   
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
    '''
    Concatenates args in order to create a filename (for pdf reporting purposes)

    Args :
        list of float, int and str

    Returns :
        str with all arguments
    ''' 
    return('_'.join([str(i) for i in args]))