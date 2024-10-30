
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

from m_functions import *

@st.cache_data
def obter_tickers_energia():
    # Tickers representativos do setor de energia
    return {
        'Exxon Mobil (XOM)': 'XOM',
        'Chevron (CVX)': 'CVX',
        'BP (BP)': 'BP',
        'TotalEnergies (TOT)': 'TTE',
        'Shell (SHEL)': 'SHEL',
        'ConocoPhillips (COP)': 'COP',
        'Equinor (EQNR)': 'EQNR',
        'Eni (ENI)': 'E',
        'Schlumberger (SLB)': 'SLB',
        'Baker Hughes (BKR)': 'BKR',
        'Crude Oil (CL=F)': 'CL=F',
        'Heating Oil (HO=F)': 'HO=F',
        'Natural Gas (NG=F)': 'NG=F',
        'RBOB Gasoline (RB=F)': 'RB=F',
        'Brent Crude Oil (BZ=F)': 'BZ=F'
    }

tickers = obter_tickers_energia()

energia_limpa = {
    'NextEra Energy (NEE)': 'NEE',
    'Vestas Wind Systems (VWDRY)': 'VWDRY',
    'Plug Power (PLUG)': 'PLUG',
    'Enphase Energy (ENPH)': 'ENPH',
    'Ormat Technologies (ORA)': 'ORA',
    'First Solar (FSLR)': 'FSLR',
    'Brookfield Renewable Partners (BEP)': 'BEP',
    'Canadian Solar (CSIQ)': 'CSIQ',
    'SunPower (SPWR)': 'SPWR',
    'SolarEdge Technologies (SEDG)': 'SEDG',
    'Renewable Energy Group (REGI)': 'REGI',
    'Invesco Solar ETF (TAN)': 'TAN',
    'Global X Renewable Energy Producers ETF (RNRG)': 'RNRG',
    'iShares Global Clean Energy ETF (ICLN)': 'ICLN',
    'First Trust NASDAQ Clean Edge Green Energy Index Fund (QCLN)': 'QCLN',
    'SPDR S&P Kensho Clean Power ETF (CNRG)': 'CNRG',
    'VanEck Vectors Low Carbon Energy ETF (SMOG)': 'SMOG'
}

cryptomoedas = {
    'Bitcoin (BTC)': 'BTC-USD',
    'Ethereum (ETH)': 'ETH-USD',
    'Ripple (XRP)': 'XRP-USD',
    'Litecoin (LTC)': 'LTC-USD',
    'Bitcoin Cash (BCH)': 'BCH-USD',
    'Cardano (ADA)': 'ADA-USD',
    'Polkadot (DOT)': 'DOT-USD',
    'Binance Coin (BNB)': 'BNB-USD',
    'Chainlink (LINK)': 'LINK-USD',
    'Stellar (XLM)': 'XLM-USD',
    'Dogecoin (DOGE)': 'DOGE-USD',
    'Solana (SOL)': 'SOL-USD',
    'Tron (TRX)': 'TRX-USD',
    'Monero (XMR)': 'XMR-USD',
    'EOS (EOS)': 'EOS-USD',
    'Tezos (XTZ)': 'XTZ-USD',
    'NEO (NEO)': 'NEO-USD',
    'VeChain (VET)': 'VET-USD',
    'Dash (DASH)': 'DASH-USD',
    'Zcash (ZEC)': 'ZEC-USD'
}

energia_nao_limpa = {
        'Exxon Mobil (XOM)': 'XOM',
        'Chevron (CVX)': 'CVX',
        'BP (BP)': 'BP',
        'TotalEnergies (TTE)': 'TTE',
        'Shell (SHEL)': 'SHEL',
        'ConocoPhillips (COP)': 'COP',
        'Equinor (EQNR)': 'EQNR',
        'Eni (ENI)': 'E',
        'Schlumberger (SLB)': 'SLB',
        'Baker Hughes (BKR)': 'BKR',
        'Crude Oil (CL=F)': 'CL=F',
        'Heating Oil (HO=F)': 'HO=F',
        'Natural Gas (NG=F)': 'NG=F',
        'RBOB Gasoline (RB=F)': 'RB=F',
        'Brent Crude Oil (BZ=F)': 'BZ=F'
    }



st.subheader("Análise Multifractal Detrended Fluctuation")

    # Selectbox para escolher um ativo
ativo = st.selectbox("Escolha um ativo", list(tickers.keys()) + ["Outro (inserir ticker)"])

# Verificar se o usuário escolheu inserir um ticker manualmente
if ativo == "Outro (inserir ticker)":
    ticker = st.text_input("Insira o ticker do ativo")
else:
    ticker = tickers[ativo]


data_inicio = st.date_input("Data de início", datetime(2010, 1, 1))
data_fim = st.date_input("Data de fim", datetime(2024, 1, 1))

if ticker:
    # Adquirir dados históricos do ativo
    data = yf.download(ticker, start=data_inicio, end=data_fim)
        # Entradas para parâmetros
    st.sidebar.subheader("Parâmetros MF-DFA")

    order = st.sidebar.number_input("Ordem do ajuste polinomial", min_value=1, max_value=5, value=1, step=1)

    min_lag_exp = st.sidebar.slider("Valor mínimo do s", min_value=10, max_value=20, value=12)
    max_lag_exp = st.sidebar.slider("Valor máximo do s", min_value=int((len(data))/10), max_value=int((len(data))/7), value=int((len(data))/8))


    if not data.empty:
        st.write(f"Dados carregados para {ativo}")
        st.write(data.tail())

        # Selecionar o preço de fechamento para análise
        #serie_temporal = data['Adj Close'].dropna().pct_change().dropna().values
        serie_temporal = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna().values


        st.divider()
        # Executar MFDFA ----------------------------------------------------------------------------------

        st.subheader("Multifractal Spectrum Analysis")

        try:
            st.write("Executando MFDFA...")
            # Definindo os valores de lag com base na entrada do usuário
            lag = np.unique(np.linspace(min_lag_exp, max_lag_exp, 34).astype(int))

            #lag = np.unique(np.logspace(0.5, 3, 100).astype(int))
            #st.write(lag)
            # Calcular o (MF)DFA
            st.write("Selecione o valores de q")# Sidebar inputs
            min_value = st.number_input("Valor mínimo", min_value=-100, max_value=0, value=-60, step=1)
            max_value = st.number_input("Valor máximo", min_value=0, max_value=100, value=60, step=1)
            step_value = st.number_input("Passo (step)", min_value=1, max_value=100, value=3, step=1)
            # Generate the array based on the inputs
            q = np.arange(min_value, max_value + step_value, step_value)

            lag, dfa = MFDFA(serie_temporal, lag=lag, q=q, order=order)

            # Visualizar os resultados
            n_series = dfa.shape[1]  # Número de séries em dfa
            plt.figure(figsize=(10, 5))
            # Loop para plotar cada série de dfa
            for i in range(0, n_series, 5):  # Incremento de 5
                plt.loglog(lag, dfa[:, i], 'o', label=f'q= {q[i+1]}')

            plt.xlabel("Ln_s")
            plt.ylabel("LnFq(s)")
            plt.title("MF-DFA Log-Log Plot")
            plt.legend()
            plt.show() 
            st.pyplot(plt)
            

        
            st.write("Hurst expoente para um q específico")
            q_value = st.number_input("Selecione o valor de q", min_value=-100, max_value=100, value=2, step=1)

            lag_value, dfa_value = MFDFA(serie_temporal, lag=lag, q=q_value, order=order)
            # Ajuste linear para estimar o índice de Hurst
            H_hat = np.polyfit(np.log(lag_value)[4:20], np.log(dfa_value)[4:20], 1)[0]

            # Exibir o valor estimado de H
            st.write(f'Estimated H = '+'{:.3f}'.format(H_hat[0]))

        except Exception as e:
            st.error(f"Erro na análise MF-DFA: {e}")
    
# ------------------------------------------------------------------------------------------
        st.divider()
        try:          
            st.subheader("Sources of Multifractality")
            # Calcular o espectro de singularidade (Singularity Spectrum)
            alpha, f_alpha = singularity_spectrum(lag, dfa, q)

            st.write(""" 
                        
Singularity spectrum                   
-------
alpha: np.array
    Singularity strength `α`. The width of this function indicates the
    strength of the multifractality. A width of `max(α) - min(α) ≈ 0`
    means the data is monofractal.

f: np.array
    Singularity spectrum `f(α)`. The location of the maximum of `f(α)`
    (with `α` as the abscissa) should be 1 and indicates the most
    prominent fractal scale in the data.
                        """)

            # Plotar o espectro de singularidade
            plt.figure(figsize=(10, 5))
            plt.plot(alpha, f_alpha, 'o-')
            plt.xlabel('α (Força de Singularity)')
            plt.ylabel('f(α) (Espectro de Singularity)')
            plt.title('Singularity Spectrum')
            st.pyplot(plt)

            # Calcular os expoentes de escalabilidade (Scaling Exponents)
            tau_q = scaling_exponents(lag, dfa, q)

            st.write(""" 
                            
Scaling exponents                  
-------                           
q: np.array
    The `q` powers.

tau: np.array
    Scaling exponents `τ(q)`. A usually increasing function of `q` from
    which the fractality of the data can be determined by its shape. A
    truly linear tau indicates monofractality, whereas a curved one
    (usually curving around small `q` values) indicates multifractality.
                        """)

            # Plotar os expoentes de escalabilidade
            plt.figure(figsize=(10, 5))
            plt.plot(tau_q[0], tau_q[1], 'o-')
            plt.xlabel('q (Ordem do Momento)')
            plt.ylabel('τ(q) (Expoente de Escalabilidade)')
            plt.title('Scaling Exponents')
            st.pyplot(plt)

            # Calcular os expoentes de Hurst generalizados (Hurst Exponents)
            hurst_q = hurst_exponents(lag, dfa, q)

            st.write(""" 
                            
Hurst exponents               
-------                                         
q: np.array
    The `q` powers.

hq: np.array
    Singularity strength `h(q)`. The width of this function indicates the
    strength of the multifractality. A width of `max(h(q)) - min(h(q)) ≈ 0`
    means the data is monofractal.
                        """)

            # Plotar os expoentes de Hurst
            plt.figure(figsize=(10, 5))
            plt.plot(hurst_q[0], hurst_q[1], 'o-')
            plt.xlabel('q (Ordem do Momento)')
            plt.ylabel('H(q) (Expoente de Hurst)')
            plt.title('Generalized Hurst Exponents')
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Erro na análise Sources of Multifractality: {e}")
    else:
        st.error("Erro ao carregar dados. Verifique o ticker selecionado ou a conectividade de rede.")

