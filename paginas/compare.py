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

st.subheader("Comparação de Ativos")
# Criação das colunas para inputs lado a lado
col1, col2 = st.columns(2)

with col1:
    st.write("**Ativo 1**")
    ativo1 = st.selectbox("Escolha o primeiro ativo", list(energia_limpa.keys()) + ["Outro (inserir ticker)"], key='ativo1')
    if ativo1 == "Outro (inserir ticker)":
        ticker1 = st.text_input("Insira o ticker do primeiro ativo")
    else:
        ticker1 = energia_limpa[ativo1]

    data_inicio1 = st.date_input("Data de início", datetime(2010, 1, 1), key='inicio1')
    data_fim1 = st.date_input("Data de fim", datetime(2024, 1, 1), key='fim1')

with col2:
    st.write("**Ativo 2**")
    ativo2 = st.selectbox("Escolha o segundo ativo", list(energia_nao_limpa.keys()) + ["Outro (inserir ticker)"], key='ativo2')
    if ativo2 == "Outro (inserir ticker)":
        ticker2 = st.text_input("Insira o ticker do segundo ativo")
    else:
        ticker2 = energia_nao_limpa[ativo2]

    data_inicio2 = st.date_input("Data de início", datetime(2010, 1, 1), key='inicio2')
    data_fim2 = st.date_input("Data de fim", datetime(2024, 1, 1), key='fim2')


if ticker1 and ticker2:
    # Adquirir dados históricos dos ativos
    data1 = yf.download(ticker1, start=data_inicio1, end=data_fim1)
    data2 = yf.download(ticker2, start=data_inicio2, end=data_fim2)

    if not data1.empty and not data2.empty:
        # Ajustar datas finais para a última data disponível
        data_inicio1 = max(data_inicio1, data1.index[0].date())
        data_inicio2 = max(data_inicio2, data2.index[0].date())

        # Rebaixar os dados com base nas datas ajustadas
        data1 = yf.download(ticker1, start=data_inicio1, end=data_fim1)
        data2 = yf.download(ticker2, start=data_inicio2, end=data_fim2)

        if not data1.empty and not data2.empty:
            st.write(f"Dados carregados para \n\n {ativo1} Data: {data_inicio1}-{data_fim1} \n\n {ativo2} Data: {data_inicio2}-{data_fim2}")

            # Selecionar o preço de fechamento ajustado para análise
            serie_temporal1 = np.log(data1['Adj Close'] / data1['Adj Close'].shift(1)).dropna().values
            serie_temporal2 = np.log(data2['Adj Close'] / data2['Adj Close'].shift(1)).dropna().values

            st.divider()
            st.subheader("Multifractal Spectrum Analysis Comparativa")

            try:
                st.write("Executando MFDFA...")

                # Entradas para parâmetros
                order = st.sidebar.number_input("Ordem do ajuste polinomial", min_value=1, max_value=5, value=1, step=1)
                min_lag_exp = st.sidebar.slider("Valor mínimo do s", min_value=10, max_value=20, value=12)
                max_lag_exp = st.sidebar.slider("Valor máximo do s", min_value=int((len(data1))/10), max_value=int((len(data1))/7), value=int((len(data1))/8))
                lag = np.unique(np.linspace(min_lag_exp, max_lag_exp, 34).astype(int))
                
                st.write("Selecione o valores de q")
                min_value = st.number_input("Valor mínimo", min_value=-100, max_value=100, value=-60, step=1)
                max_value = st.number_input("Valor máximo", min_value=-100, max_value=100, value=60, step=1)
                step_value = st.number_input("Passo (step)", min_value=1, max_value=100, value=3, step=1)
                q = np.arange(min_value, max_value + step_value, step_value)

                # Calcular o (MF)DFA para ambos os ativos
                lag1, dfa1 = MFDFA(serie_temporal1, lag=lag, q=q, order=order)
                lag2, dfa2 = MFDFA(serie_temporal2, lag=lag, q=q, order=order)

                col1, col2 = st.columns(2)
                with col1:
                    # Visualizar os resultados comparativos
                    plt.figure(figsize=(10, 5))
                    for i in range(0, dfa1.shape[1], 5):  
                        plt.loglog(lag1, dfa1[:, i], 'o-', label=f'{ativo1} - q= {q[i+1]}')

                    plt.xlabel("Ln_s")
                    plt.ylabel("LnFq(s)")
                    plt.title("MF-DFA Log-Log Plot Comparativo")
                    plt.legend()
                    st.pyplot(plt)

                with col2:
                    # Visualizar os resultados comparativos
                    plt.figure(figsize=(10, 5))
                    for i in range(0, dfa1.shape[1], 5):  
                        plt.loglog(lag2, dfa2[:, i], 'o--', label=f'{ativo2} - q= {q[i+1]}')

                    plt.xlabel("Ln_s")
                    plt.ylabel("LnFq(s)")
                    plt.title("MF-DFA Log-Log Plot Comparativo")
                    plt.legend()
                    st.pyplot(plt)



                st.write("Hurst expoente para um q específico")
                q_value = st.number_input("Selecione o valor de q", min_value=-100, max_value=100, value=2, step=1)

                lag_value, dfa_value = MFDFA(serie_temporal1, lag=lag, q=q_value, order=order)
                lag_value2, dfa_value2 = MFDFA(serie_temporal2, lag=lag, q=q_value, order=order)

                # Ajuste linear para estimar o índice de Hurst
                H_hat = np.polyfit(np.log(lag_value)[4:20], np.log(dfa_value)[4:20], 1)[0]
                H_hat2 = np.polyfit(np.log(lag_value2)[4:20], np.log(dfa_value2)[4:20], 1)[0]

                # Exibir o valor estimado de H
                st.write(f'Estimated H of {ticker1}= '+'{:.3f}'.format(H_hat[0]))
                st.write(f'Estimated H of {ticker2}= '+'{:.3f}'.format(H_hat2[0]))



                st.divider()
                st.subheader("Comparação de Singularity Spectrum")
                alpha1, f_alpha1 = singularity_spectrum(lag1, dfa1, q)
                alpha2, f_alpha2 = singularity_spectrum(lag2, dfa2, q)

                plt.figure(figsize=(10, 5))
                plt.plot(alpha1, f_alpha1, 'o-', label=f'{ativo1}')
                plt.plot(alpha2, f_alpha2, 'o--', label=f'{ativo2}')
                plt.xlabel('α (Força de Singularity)')
                plt.ylabel('f(α) (Espectro de Singularity)')
                plt.title('Singularity Spectrum Comparativo')
                plt.legend()
                st.pyplot(plt)

                st.divider()
                st.subheader("Comparação de Scaling Exponents")
                tau_q1 = scaling_exponents(lag1, dfa1, q)
                tau_q2 = scaling_exponents(lag2, dfa2, q)

                plt.figure(figsize=(10, 5))
                plt.plot(tau_q1[0], tau_q1[1], 'o-', label=f'{ativo1}')
                plt.plot(tau_q2[0], tau_q2[1], 'o--', label=f'{ativo2}')
                plt.xlabel('q (Ordem do Momento)')
                plt.ylabel('τ(q) (Expoente de Escalabilidade)')
                plt.title('Scaling Exponents Comparativo')
                plt.legend()
                st.pyplot(plt)

                st.divider()
                st.subheader("Comparação de Generalized Hurst Exponents")
                hurst_q1 = hurst_exponents(lag1, dfa1, q)
                hurst_q2 = hurst_exponents(lag2, dfa2, q)

                plt.figure(figsize=(10, 5))
                plt.plot(hurst_q1[0], hurst_q1[1], 'o-', label=f'{ativo1}')
                plt.plot(hurst_q2[0], hurst_q2[1], 'o--', label=f'{ativo2}')
                plt.xlabel('q (Ordem do Momento)')
                plt.ylabel('H(q) (Expoente de Hurst)')
                plt.title('Generalized Hurst Exponents Comparativo')
                plt.legend()
                st.pyplot(plt)

            except Exception as e:
                st.error(f"Erro na análise comparativa: {e}")

    else:
        st.error("Erro ao carregar dados. Verifique os tickers selecionados ou a conectividade de rede.")

