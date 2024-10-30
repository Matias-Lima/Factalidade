import streamlit as st
import pandas as pd
import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import altair as alt
from MFDFA import MFDFA
import yfinance as yf
from datetime import datetime
from functions import singularity_spectrum, scaling_exponents, hurst_exponents
plt.style.use('dark_background')
from mf_adcca import dcca, basic_dcca
from arch import arch_model



# Funções -------------------------------

# Função para calcular o R/S rescaled_range_analysis
def rescaled_range_analysis(ts):
    ts = list(ts)
    N = len(ts)
    if N < 20:
        raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

    max_k = int(np.floor(N/2))
    R_S_dict = []
    for k in range(10,max_k+1):
        R,S = 0,0
        # split ts into subsets
        subset_list = [ts[i:i+k] for i in range(0,N,k)]
        if np.mod(N,k)>0:
            subset_list.pop()
            #tail = subset_list.pop()
            #subset_list[-1].extend(tail)
        # calc mean of every subset
        mean_list=[np.mean(x) for x in subset_list]
        for i in range(len(subset_list)):
            cumsum_list = pd.Series(subset_list[i]-mean_list[i]).cumsum()
            R += max(cumsum_list)-min(cumsum_list)
            S += np.std(subset_list[i])
        R_S_dict.append({"R":R/len(subset_list),"S":S/len(subset_list),"n":k})
    
    log_R_S = []
    log_n = []
    print(R_S_dict)
    for i in range(len(R_S_dict)):
        R_S = (R_S_dict[i]["R"]+np.spacing(1)) / (R_S_dict[i]["S"]+np.spacing(1))
        log_R_S.append(np.log(R_S))
        log_n.append(np.log(R_S_dict[i]["n"]))

    Hurst_exponent = np.polyfit(log_n,log_R_S,1)[0]
    return Hurst_exponent

# Função para calcular a volatilidade
def calculate_volatility(returns):
    return returns.rolling(window=21).std() * np.sqrt(252)  # Volatilidade anualizada com uma janela de 21 dias

# Função para calcular o retorno
def calculate_returns(prices):
    return prices.pct_change().dropna()

# Função para calcular a volatilidade com o modelo GARCH (1,1)
def calculate_garch_volatility(data):
    petr_gm = arch_model(data, p=1, q=1, mean='constant', vol='GARCH', dist='normal')
    petr_fit = petr_gm.fit(disp='off')
    return petr_fit.conditional_volatility

# Função para calcular a volatilidade com o modelo TGARCH (1,1,2)
def calculate_tgarch_volatility(data):
    model = arch_model(data, mean="Zero", p=1, o=1, q=1, power=1.0)
    result = model.fit(disp="off")
    return result.conditional_volatility

# Função para plotar gráficos
def plot_metrics(prices, returns, volatility, asset_name):
    st.subheader(f"Gráficos para {asset_name}")

    st.write("**Preço**")
    st.line_chart(prices)

    st.write("**Retorno Diário**")
    st.line_chart(returns)

    st.write("**Volatilidade Anualizada (21 dias)**")
    st.line_chart(volatility)

def plot_dcca(dat1, dat2):

    st.subheader("Análise Multifractal DCCA")

    # Configurando a figura no Streamlit
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    # Overall
    q_values = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    colors = ['#9467bd'] * 11
    for q, color in zip(q_values, colors):
        plot_df = np.log(np.reshape(basic_dcca(dat1, dat2, Q=[q])[:2], [2, 100]))
        ax[0].plot(plot_df[0], plot_df[1], linestyle='solid', color=color)
    ax[0].set_xlabel('$\ln(s)$', size=12)
    ax[0].set_ylabel('$\ln(F_q(s))$', size=12)
    ax[0].set_title('Price-Volatility (Overall)')
    ax[0].grid(True)
    ax[0].legend(['q = {}'.format(q) for q in q_values])

    # Uptrend
    for q, color in zip(q_values, ['#1f77b4'] * 11):
        plot_df = np.log(np.reshape(basic_dcca(dat1, dat2, Q=[q])[3:5], [2, 100]))
        ax[1].plot(plot_df[0], plot_df[1], linestyle='dashed', color=color)
    ax[1].set_xlabel('$\ln(s)$', size=12)
    ax[1].set_ylabel('$\ln(F^+_q(s))$', size=12)
    ax[1].set_title('Price-Volatility (Uptrend)')
    ax[1].grid(True)
    ax[1].legend(['q = {}'.format(q) for q in q_values])

    # Downtrend
    for q, color in zip(q_values, ['#d62728'] * 11):
        plot_df = np.log(np.reshape(basic_dcca(dat1, dat2, Q=[q])[6:8], [2, 100]))
        ax[2].plot(plot_df[0], plot_df[1], linestyle='dashdot', color=color)
    ax[2].set_xlabel('$\ln(s)$', size=12)
    ax[2].set_ylabel('$\ln(F^-_q(s))$', size=12)
    ax[2].set_title('Price-Volatility (Downtrend)')
    ax[2].grid(True)
    ax[2].legend(['q = {}'.format(q) for q in q_values])

    # Exibir gráfico no Streamlit
    st.pyplot(fig)

# Função para plotar A-MFDFA e MF-ADCCA
def plot_amfdfa_mfadcca(dat1, dat2, trend_base):

    st.subheader("A-MFDFA e MF-ADCCA para Price-Volatility")

    # Tamanho da figura
    fig = plt.figure(figsize=(6, 4))

    # Definindo labels dos eixos
    plt.xlabel('$q$', size=14)
    plt.ylabel('Exponentes de Hurst generalizados', size=14)

    # Valores de q a serem plotados
    qorders = list(range(-60, 60))

    # Cálculo DCCA
    est_results_xy = basic_dcca(dat1, dat2, Q=qorders, trend_base=trend_base, asymmetry_base='optional')

    # Plotagem dos resultados
    plt.plot(qorders, est_results_xy[2], marker="o", markersize=7, linestyle='solid', label='overall', color='#9467bd')
    plt.plot(qorders, est_results_xy[5], marker="^", markersize=7, linestyle='solid', label='uptrend', color='#1f77b4')
    plt.plot(qorders, est_results_xy[8], marker="v", markersize=7, linestyle='solid', label='downtrend', color='#d62728')

    # Limites do eixo y
    #plt.ylim(0, 1)

    # Exibir legenda
    plt.legend(frameon=True)

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

# Função para calcular e plotar a função massa tau(q)
def plot_mass_function_tau_q(dat1, dat2, trend_base):

    st.subheader("Função Massa Tau(q) usando MFDFA e MFDCCA")

    # Tamanho da figura
    fig = plt.figure(figsize=(6, 4))

    # Definindo labels dos eixos
    plt.xlabel('$q$', size=14)
    plt.ylabel('Função Massa', size=14)

    # Valores de q a serem plotados
    qorders = list(range(-10, 11))

    # Cálculo DCCA
    est_results_xy = basic_dcca(dat1, dat2, Q=qorders, trend_base=trend_base, asymmetry_base='optional')

    # Exponentes de Hurst generalizados
    gh_xy_overall = est_results_xy[2]
    gh_xy_uptrend = est_results_xy[5]
    gh_xy_downtrend = est_results_xy[8]

    # Plot da função massa tau(q)
    plt.plot(qorders, gh_xy_overall * np.array(qorders) - 1, marker="o", markersize=7, linestyle='solid', label='overall', color='#9467bd')
    plt.plot(qorders, gh_xy_uptrend * np.array(qorders) - 1, marker="^", markersize=7, linestyle='solid', label='uptrend', color='#1f77b4')
    plt.plot(qorders, gh_xy_downtrend * np.array(qorders) - 1, marker="v", markersize=7, linestyle='solid', label='downtrend', color='#d62728')

    # Exibir legenda
    plt.legend(frameon=True)

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

# Função para calcular o espectro de singularidade
def plot_singularity_spectra(dat1, dat2, trend_base):
    fig_2 = plt.figure(figsize=(6, 4))  # Tamanho da figura

    # Definindo labels dos eixos
    plt.xlabel('$alpha$', size=14)
    plt.ylabel('Singularity Spectra', size=14)

    # Valores de q a serem plotados (entre -5 e 5)
    qorders = np.array([-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]).tolist()
    qorders_D = (np.array(qorders) + 0.1).tolist()

    # Cálculo DCCA
    est_results_xy = basic_dcca(dat1, dat2, Q=qorders, trend_base=trend_base, asymmetry_base='optional')
    est_results_Dxy = basic_dcca(dat1, dat2, Q=qorders_D, trend_base=trend_base, asymmetry_base='optional')

    # Cálculo de tau(q)
    tau_xy_overall = est_results_xy[2] * np.array(qorders) - 1
    tau_xy_uptrend = est_results_xy[5] * np.array(qorders) - 1
    tau_xy_downtrend = est_results_xy[8] * np.array(qorders) - 1

    tau_Dxy_overall = est_results_Dxy[2] * np.array(qorders_D) - 1
    tau_Dxy_uptrend = est_results_Dxy[5] * np.array(qorders_D) - 1
    tau_Dxy_downtrend = est_results_Dxy[8] * np.array(qorders_D) - 1

    # Cálculo de alpha (derivada numérica de tau(q))
    alpha_xy_overall = (tau_Dxy_overall - tau_xy_overall) / 0.1
    alpha_xy_uptrend = (tau_Dxy_uptrend - tau_xy_uptrend) / 0.1
    alpha_xy_downtrend = (tau_Dxy_downtrend - tau_xy_downtrend) / 0.1

    # Plotando o espectro de singularidade
    plt.plot(alpha_xy_overall, np.array(qorders) * alpha_xy_overall - tau_xy_overall, marker="o", markersize=7, linestyle='solid', label='overall', color='#9467bd')
    plt.plot(alpha_xy_uptrend, np.array(qorders) * alpha_xy_uptrend - tau_xy_uptrend, marker="^", markersize=7, linestyle='solid', label='uptrend', color='#1f77b4')
    plt.plot(alpha_xy_downtrend, np.array(qorders) * alpha_xy_downtrend - tau_xy_downtrend, marker="v", markersize=7, linestyle='solid', label='downtrend', color='#d62728')

    # Exibir legenda e ajustar os limites do gráfico
    plt.legend(frameon=True)

    # Exibir o gráfico
    st.pyplot(fig_2)

