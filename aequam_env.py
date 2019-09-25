'''
Class to adapt Open AI gym environments to Aequam Capital specific's problematic
'''


import gym
from gym import spaces

import warnings
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from backtest_utils import *

REPORT_PATH = 'C:/Users/Avisia/Documents/Dorian/Gym_RL/Aequam_v0/reports/'

class AequamEnv(gym.Env):
    '''
    Trading environment class inheriting from gym.Env

    Methods:
        __init__
        step
        reset
        render
        play_last_episode
        plot_last_episode
        plot_positions
        print_pdf_report
        _next_observation
    '''

    metadata = {'render.modes': ['human', 'live', 'file']}
    visualization = None

    def __init__(self, df_obs, df_prices, lookback_window = 5, transaction_cost = 0.001, starting_cash = 100.0, \
                max_balance = 1000.0, reward_type='delayed', transaction_smoothing = 10, reward_window = 10,\
                risk_aversion = 5, report_path = REPORT_PATH):

        '''
        Args:
            df_obs (pandas.DataFrame): the DataFrame that contains potentially explaining data (signal-providing)
            df_prices (pandas.DataFrame): the DataFrame that contains prices of the assets the agent can trade (one column per asset)
            lookback_window (int): the amount of rows the agent can observe in the past
            transaction_cost (float): backtest hypothesis on the percentage of transaction cost
            starting_cash (float): the amount of cash to start the strategy
            max_balance (float): the maximum balance the agent can hold (estimation, to feed the algorithm)
            reward_type (str in ['delayed','daily','vol']): the reward philosophy for the agent. Other philosophies can be added
            transaction_smoothing (float): penalty for high number of transactions. Useful for 'vol' reward
            reward_window (int): time parameter of the strategy. The higher it is, the more long-term the strategy is. Useful for 'vol' reward
            risk_aversion (float): penalty for high drawdown. Useful for 'vol' reward
            report_path (str): path to the folder you want to save the pdf reports in

        Attributes:
            n_observations (int): the number of columns in df_obs (the number of explicative variables)
            n_assets (int): the number of columns in df_prices (the number of assets or portfolios the agent can trade)
            total_window (int): the number of rows in df_obs (to see the number of trading days)
            df_render (pandas.DataFrame): a DataFrame containing useful information on the portfolio traded across time
            action_space (MANDATORY FOR GYM, gym.spaces): the space of action. Can be Discrete or Box (continuous)
            observation_space (MANDATORY FOR GYM, gym.spaces): the space of observations. Can be Discrete or Box (continuous)
            WARNING : the types of action_space and observation_space (Discrete, Box, MultiDiscrete, ...) limits the type of model the agent can use
        '''
  
        super(AequamEnv, self).__init__()
    
        #.....
        #More initialisation parameters
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.total_tc = 0.0 #total transaction costs (keep track)
        self.starting_cash = starting_cash
        self.timestep = self.lookback_window -2
        self.pf_value = starting_cash
        self.last_action = 0
        self.reward_range = (0, max_balance)*(reward_type == 'delayed') + (-2,2)*(reward_type == 'daily') + (-2,2)*(reward_type == 'vol') #for agent
        self.reward_type = reward_type
        self.transaction_smoothing = transaction_smoothing
        self.reward_window = reward_window
        self.risk_aversion = risk_aversion
        #.....

        self.df_obs = df_obs
        self.df_prices = rescale(df_prices, self.lookback_window-2, self.starting_cash) #start with same value at the beginning
        self.df_prices['Cash'] = self.starting_cash
        self.df_render = self.df_prices.copy()
        
        self.n_observations = len(self.df_obs.columns)
        self.n_assets = len(self.df_prices.columns)
        self.total_window = len(self.df_obs)        
    
        self.df_render['Equally_weighted'] = self.df_prices.iloc[:,:self.n_assets].mean(axis=1)
        self.df_render['Action']= self.last_action
        self.df_render['Transaction_cost']=0.0
        self.df_render['Portfolio_value']=self.pf_value

        #For a start, at each time we select one of the portfolio
        self.action_space = spaces.Discrete(self.n_assets)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=np.tile(np.array(self.df_obs.min()),(self.lookback_window,1)), \
                                            high=np.tile(np.array(self.df_obs.max()), (self.lookback_window,1)),\
                                            dtype=np.float16)

        self.report_path = report_path
        
    def _next_observation(self):
        '''
        Goes one step in time and change the observation for the agent
        '''
        return(np.array(self.df_obs.iloc[(self.timestep-self.lookback_window+2):(self.timestep +2),:]))

    def step(self, action):
        #Set next observation
        obs = self._next_observation()
        
        #Fill render DataFrame and run one timestep
        new_tc = self.pf_value * self.transaction_cost * (1 - (action == self.last_action))
        self.total_tc += new_tc
        
        #store
        a=self.pf_value *1
        #update portfolio values
        self.pf_value *= (self.df_render.iloc[self.timestep+1,action]/self.df_render.iloc[self.timestep,action])
        self.pf_value -= new_tc
        
        self.df_render.iloc[self.timestep+1,-3:] = [action, self.total_tc, self.pf_value]
        
        self.timestep += 1
        
        #Three types of rewards: more can be added and fine-tuned
        if self.reward_type == 'delayed': #dummy one, found on Github for similar project
            delay_modifier = (self.timestep / self.total_window)
            reward = self.pf_value * delay_modifier
        elif self.reward_type == 'daily': #daily reward corresponding for daily return penalized by transaction cost
            reward = self.pf_value/a * 100 - self.transaction_smoothing * self.total_tc
        elif self.reward_type == 'vol': #ultimate one, which was designed and validated by research team
            reward_pf = np.array(self.df_render.iloc[self.timestep:(self.timestep+self.reward_window), action])
            reward_returns = np.diff(reward_pf) / reward_pf[1:] 
            return_metric = (reward_pf[-1]-reward_pf[0])/reward_pf[0]
            transaction_cost_metric = self.transaction_cost * self.pf_value/self.starting_cash * (1-(self.last_action == action)*1)
            semivariance = reward_returns[reward_returns<0].std()
            if np.isnan(semivariance):
                semivariance = 0.0
            #reward is return on a frozen portfolio for a number of days, penalized by semivariance and transaction costs
            reward = return_metric - self.risk_aversion * semivariance - self.transaction_smoothing * transaction_cost_metric
            
        else:
            raise ValueError
        
        self.last_action = int(action)
        #Set the counter to done or not
        done = (self.timestep > len(self.df_obs)-2 - self.reward_window) | (self.pf_value < 0)
        
        #Add additional info for debugging purposes
        info = {}
        
        return obs, reward, done, info
    
    def reset(self):
        
        self.total_tc = 0.0
        self.starting_cash = self.starting_cash
        self.timestep = self.lookback_window -2
        self.pf_value = self.starting_cash
        self.last_action = 0
        
        self.df_render = self.df_prices.copy()
        self.df_render['Equally_weighted'] = self.df_prices.iloc[:,:self.n_assets].mean(axis=1)
        self.df_render['Action']= self.last_action
        self.df_render['Transaction_cost']=0.0
        self.df_render['Portfolio_value']=self.pf_value    

        return(self._next_observation())
        
    def render(self):
        pass

    def play_last_episode(self, model):
        '''
        Given a model, play the last episode and stop before reset

        Args:
            model: a model within stable_baselines, already initialized with AequamEnv


        '''
        obs = self.reset()
        for i in range(self.total_window -self.lookback_window-1):
            action, _states = model.predict(obs)
            obs, rewards, done, info = self.step(action)


    def plot_last_episode(self, show=True, save=False):
        '''
        Plots the terminal portfolio value for each episode, to see the learning process
        
        Args:
            show (bool): show the graph
            save (bool): save the graph in a picture
        '''
        plt.figure(figsize=(15,7))
        df_to_plot = self.df_render.iloc[(self.lookback_window-2):-2,:][['Portfolio_value', 'Equally_weighted']]   
        df_to_plot['diff'] = df_to_plot['Portfolio_value']/df_to_plot['Equally_weighted']

        df_to_plot[['Equally_weighted', 'Portfolio_value']].plot()
        df_to_plot['diff'].plot(secondary_y = True, style = '--')

        plt.title('What was the last episode like?')
        plt.xlabel('Date')
        plt.ylabel('Value')
        if save:
            plt.savefig(self.report_path+'plot_last_episode.png', dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_positions(self, show=True, save=False):
        '''
        Plots the terminal portfolio value with positions at all times
        
        Args:
            show (bool): show the graph
            save (bool): save the graph in a picture
        '''
        plt.figure(figsize=(15,7))
        self.df_render['Action'].plot()
        percent = np.sum((self.df_render['Action'].diff().dropna()!=0)*1)\
                 / (self.total_window -1 - self.lookback_window)*100
        plt.title('What were the positions? Traded {} percent of time'.format(np.round(percent,1)))
        plt.xlabel('Date')
        plt.ylabel('Position')
        if save:
            plt.savefig(self.report_path+'plot_positions.png', dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

    def print_pdf_report(self, title='sans_titre'):
        '''
        Print a pdf report containing all graphics. Useful for quick visual analysis

        Arg:
            title (str): The title of the pdf file

        Returns:
            'Done', meaning the export is done

        '''
        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font('arial', 'B', 12)

        self.plot_last_episode(show=False, save=True)
        self.plot_positions(show=False, save=True)

        pdf.image(self.report_path+'plot_last_episode.png', x = None, y = None, w = 150, h = 0, type = '', link = '')
        pdf.image(self.report_path+'plot_positions.png', x = None, y = None, w = 150, h = 0, type = '', link = '')
        
        pdf.output(self.report_path+title+'.pdf', 'F')
        return('Done')