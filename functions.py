import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBClassifier

import streamlit as st
import plotly.express as px
from streamlit_js_eval import streamlit_js_eval


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> app functions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def info_msg(text):
    st.info(text, icon="ℹ")

def clean_corr_vars():
    st.session_state.show_corr_map = None
    st.session_state.corr_map = None
    st.session_state.corr_response = None

def clean_model_vars():
    st.session_state.last_model = None
    st.session_state.last_model_uplift = None

def clean_vars():
    clean_corr_vars()
    clean_model_vars()

def file_to_df(data_file):
    if data_file.name.split('.')[1] == 'xlsx':
        return pd.read_excel(data_file)
    elif data_file.name.split('.')[1] == 'parquet':
        return pd.read_parquet(data_file)
    elif data_file.name.split('.')[1] == 'csv':
        return pd.read_csv(data_file)
    
def data_info(data):
    st.write('Розмір даних:', data.shape)
    st.write('Даних:', data.shape)
    st.dataframe(data)
    dtypes = pd.DataFrame(data.dtypes).T
    dtypes.index = ['dtype']
    st.write('Статистичні характеристики даних:')
    st.dataframe(pd.concat([dtypes, data.describe(include='all').round(4)]).astype(str))

def corr_heatmap(data):
    corr = data.corr(method='spearman').round(3)
    # window_width = streamlit_js_eval(js_expressions='window.innerWidth', key = 'SCR')
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    # fig.update_layout(autosize=False, width=window_width)
    return fig

def get_x_y_t(data, id_col, y_col, t_col):
    df = data.drop(columns=id_col)
    return df.drop(columns=[t_col, y_col]), df[y_col], df[t_col]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> uplift functions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>> Model
class TwoModelsDDR(BaseEstimator):
    def __init__(self, classifier_trmt, classifier_ctrl, ddr_feature='treatment'):
        self.classifier_trmt = clone(classifier_trmt)
        self.classifier_ctrl = clone(classifier_ctrl)
        if ddr_feature in ('treatment', 'control'):
            self.ddr_feature = ddr_feature
        else:
            raise TypeError("Only 'treatment' or 'control' values are allowed in ddr_feature")


    def fit(self, X, y, t):
        X, y = check_X_y(X, y)
        t = check_array(t, ensure_2d=False)

        X_trmt = np.copy(X[t==1])
        y_trmt = np.copy(y[t==1])
        X_ctrl = np.copy(X[t==0])
        y_ctrl = np.copy(y[t==0])

        if self.ddr_feature == 'treatment':
            self.classifier_trmt.fit(X_trmt, y_trmt)

            trmt_prob = self.classifier_trmt.predict_proba(X_ctrl)[:, 1].reshape(-1, 1)
            X_ctrl = np.concatenate((X_ctrl, trmt_prob), axis=1)
            self.classifier_ctrl.fit(X_ctrl, y_ctrl)

        elif self.ddr_feature == 'control':
            self.classifier_ctrl.fit(X_ctrl, y_ctrl)

            ctrl_prob = self.classifier_ctrl.predict_proba(X_trmt)[:, 1].reshape(-1, 1)
            X_trmt = np.concatenate((X_trmt, ctrl_prob), axis=1)
            self.classifier_trmt.fit(X_trmt, y_trmt)

        else:
            raise TypeError("Only 'treatment' or 'control' values are allowed in ddr_feature")
        
        return self
    

    def predict(self, X):
        check_is_fitted(self.classifier_trmt)
        check_is_fitted(self.classifier_ctrl)
        X = check_array(X)

        if self.ddr_feature == 'treatment':
            trmt_prob = self.classifier_trmt.predict_proba(X)[:, 1]
            X = np.concatenate((X, trmt_prob.reshape(-1, 1)), axis=1)
            ctrl_prob = self.classifier_ctrl.predict_proba(X)[:, 1]

        elif self.ddr_feature == 'control':
            ctrl_prob = self.classifier_ctrl.predict_proba(X)[:, 1]
            X = np.concatenate((X, ctrl_prob.reshape(-1, 1)), axis=1)
            trmt_prob = self.classifier_trmt.predict_proba(X)[:, 1]
        else:
            raise TypeError("Only 'treatment' or 'control' values are allowed in ddr_feature")

        return trmt_prob - ctrl_prob
    

    def __sklearn_is_fitted__(self):
        return 

# >>>>> Metrics
def uplift_at_top_perc(treatment_flg, response, uplift, top_perc=0.3):
    df = pd.DataFrame([[val1, val2, val3] for val1, val2, val3 in zip(list(treatment_flg), list(response), list(uplift))], columns=['treatment_flg', 'response', 'uplift'])
    df = df.sort_values(by='uplift', ascending=False)
    df = df.iloc[:int(df.shape[0] * top_perc), :].copy()

    return df[df['treatment_flg']==1]['response'].mean() - df[df['treatment_flg']==0]['response'].mean()

def uplift_by_percentile_table(treatment_flg, response, uplift):
    tmp = pd.DataFrame([[val1, val2, val3] for val1, val2, val3 in zip(list(treatment_flg), list(response), list(uplift))], columns=['treatment_flg', 'response', 'uplift'])
    tmp['percentile'] = pd.qcut(tmp['uplift'], 10, labels=[f'{i*10} - {(i+1)*10}' for i in range(9, -1, -1)])

    tmp2 = tmp.groupby(['percentile', 'treatment_flg']).agg(
        n = ('response', 'count'), 
        avg_treatment = ('response', 'mean')
    ).reset_index()

    ubp_table = tmp2.pivot(index='percentile', columns='treatment_flg')[['n', 'avg_treatment']]
    ubp_table.columns = [f'{a}_{b}' for a in ['n', 'response_rate'] for b in ['control', 'treatment']]
    ubp_table = ubp_table[[f'{a}_{b}' for a in ['n', 'response_rate'] for b in ['treatment', 'control']]].copy()
    ubp_table = ubp_table.reset_index().sort_values(by='percentile', ascending=False)
    ubp_table['uplift'] = ubp_table['response_rate_treatment'] - ubp_table['response_rate_control']

    return ubp_table

def plot_uplift_by_percentile_barchart(treatment_flg, response, uplift, top_perc=0.3):
    ubp_table = uplift_by_percentile_table(treatment_flg, response, uplift)
    uatp = uplift_at_top_perc(treatment_flg, response, uplift, top_perc=top_perc)
    wau = (ubp_table['n_treatment'] * ubp_table['uplift']).sum() / ubp_table['n_treatment'].sum()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.set_tight_layout(True)
    width = 0.34
    r = np.arange(ubp_table.shape[0])
    
    ax1.set_title(f'Uplift by percentile\nUplift at top {top_perc:.0%} = {uatp:.2%}\nWeighted average uplift = {wau:.2%}')
    ax1.bar(r, ubp_table['uplift'], color = 'tab:green', 
            width = width, label='Uplift') 
    ax1.axhline(color='black', linewidth=.1)
    
    ax2.set_title('Response rate by percentile')
    ax2.bar(r-width/2, ubp_table['response_rate_treatment'], color = 'tab:blue', 
            width = width, label='Treatment response rate') 
    ax2.bar(r+width/2, ubp_table['response_rate_control'], color = 'tab:orange', 
            width = width, label='Control response rate') 
    
    for ax in (ax1, ax2):
        ax.set_xticks(r, ubp_table['percentile'], rotation=45) 
    
    plt.xlabel('Перцентиль')

    return fig


def calc_qini_curve(treatment_flg, response, uplift, normalize_n=True, return_uplift=False):
    df = pd.DataFrame([[val1, val2, val3] for val1, val2, val3 in zip(list(treatment_flg.astype(bool)), list(response), list(uplift))], columns=['treatment_flg', 'response', 'uplift'])
    df = df.sort_values(by='uplift', ascending=False)

    df['y_t'] = (df['response'] & df['treatment_flg']).astype(int).cumsum()
    df['y_c'] = (df['response'] & ~df['treatment_flg']).astype(int).cumsum()
    df['n_t'] = (df['treatment_flg']).astype(int).cumsum()
    df['n_c'] = (~df['treatment_flg']).astype(int).cumsum()

    df['qini'] = df.apply(lambda row: row['y_t'] - row['y_c'] * row['n_t'] / max(row['n_c'], 1), axis=1)

    df = df.reset_index(drop=True)
    df.index.name = 'n'
    df = df.reset_index()
    df['n'] = df['n'] + 1

    df = pd.concat([pd.DataFrame([[0, 0]], columns=['n', 'qini']), df])

    if normalize_n:
        df['n'] = df['n'] / df['n'].max()
    
    if return_uplift:
        return df[['n', 'qini', 'uplift']]

    return df[['n', 'qini']]

def calc_perfect_qini_curve(treatment_flg, response, normalize_n=True, return_uplift=False):
    return calc_qini_curve(treatment_flg, response, response * treatment_flg - response * (1 - treatment_flg), normalize_n=normalize_n, return_uplift=return_uplift)

def plot_qini_curve(treatment_flg, response, uplift, normalize_n=True, plot_perfect=False):
    model_curve = calc_qini_curve(treatment_flg, response, uplift, normalize_n=normalize_n)
    perfect_curve = calc_perfect_qini_curve(treatment_flg, response, normalize_n=normalize_n)
    random_curve = perfect_curve.iloc[[0, -1], :]
    
    model_auc = np.trapz(model_curve['qini'], model_curve['n'])
    perfect_auc = np.trapz(perfect_curve['qini'], perfect_curve['n'])
    random_auc = np.trapz(random_curve['qini'], random_curve['n'])
    auqc = (model_auc - random_auc) / (perfect_auc - random_auc)

    fig, ax = plt.subplots()
    ax.plot(random_curve['n'], random_curve['qini'], label='Random', color='red')
    ax.plot(model_curve['n'], model_curve['qini'], label='Model', color='blue')

    if plot_perfect:
        ax.plot(perfect_curve['n'], perfect_curve['qini'], label='Perfect', color='green')

    ax.set_title(f'Qini curve\nAUQC = {auqc:.4f}')
    if normalize_n:
        ax.set_xlabel('Відсоток прокомунікованих користувачів')
    else:
        ax.set_xlabel('Кількість прокомунікованих користувачів')
    ax.set_ylabel('Приріст виконаних цільовий дій')
    ax.legend()

    return fig

