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


st.subheader("Multifractal Asymmetric Detrended Cross-Correlation Analysis (MF-ADCCA)")

st.write("Explorando correlações cruzadas multifractais assimétricas de preço e volatilidade.")

ativo_1 = st.selectbox("Escolha o primeiro ativo", list(cryptomoedas.keys()) + ["Outro (inserir ticker)"])
ativo_2 = st.selectbox("Escolha o segundo ativo", list(cryptomoedas.keys()) + ["Outro (inserir ticker)"])

# Verificar se o usuário escolheu inserir um ticker manualmente para o primeiro ativo
if ativo_1 == "Outro (inserir ticker)":
    ticker_1 = st.text_input("Insira o ticker do primeiro ativo")
else:
    ticker_1 = cryptomoedas[ativo_1]

# Verificar se o usuário escolheu inserir um ticker manualmente para o segundo ativo
if ativo_2 == "Outro (inserir ticker)":
    ticker_2 = st.text_input("Insira o ticker do segundo ativo")
else:
    ticker_2 = cryptomoedas[ativo_2]

data_inicio = st.date_input("Data de início", datetime(2020, 1, 1))
data_fim = st.date_input("Data de fim", datetime(2024, 1, 1))

st.sidebar.write("Defina os valores de q")

min_value = st.sidebar.number_input("Valor mínimo", min_value=-100, max_value=100, value=-60, step=1)
max_value = st.sidebar.number_input("Valor máximo", min_value=-100, max_value=100, value=60, step=1)
step_value = st.sidebar.number_input("Passo (step)", min_value=1, max_value=100, value=3, step=1)
# Generate the array based on the inputs
q = np.arange(min_value, max_value + step_value, step_value)

if ticker_1 and ticker_2:

    data_1 = yf.download(ticker_1, start=data_inicio, end=data_fim)
    data_2 = yf.download(ticker_2, start=data_inicio, end=data_fim)

    if not data_1.empty and not data_2.empty:

        st.write(f"Dados carregados para {ativo_1} e {ativo_2}")

        st.write("Executando análise MF-ADCCA...")

        min_lag_exp = st.sidebar.slider("Valor mínimo do s", min_value=10, max_value=20, value=12)
        max_lag_exp = st.sidebar.slider("Valor máximo do s", min_value=int((len(data_1))/10), max_value=int((len(data_1))/7), value=int((len(data_1))/8))

        lag = np.unique(np.linspace(min_lag_exp, max_lag_exp, 34).astype(int))

        col1, col2 = st.columns(2)
        with col1:

            st.write(data_1.tail())

            serie_temporal_1 = data_1['Close'].dropna()
            st.title("Análise de Criptomoedas")
            prices = serie_temporal_1
            petr = calculate_returns(prices)
            returns = petr

            # Configuração do Streamlit
            st.title("Escolha o Modelo de Volatilidade")

            # Opções de modelo
            model_choice = st.selectbox("Selecione o modelo para cálculo de volatilidade", ["GARCH", "TGARCH"])

            # Exibir dados
            st.write("Dados dos retornos:")
            st.line_chart(petr)

            # Cálculo e exibição do resultado com base no modelo escolhido
            if model_choice == "GARCH":
                st.write("Volatilidade Condicional - Modelo GARCH:")
                vol_garch = calculate_garch_volatility(petr)
                st.line_chart(vol_garch)

            elif model_choice == "TGARCH":
                st.write("Volatilidade Condicional - Modelo TGARCH:")
                vol_tgarch = calculate_tgarch_volatility(petr)
                st.line_chart(vol_tgarch)
                
                st.write(len(vol_tgarch))

            # Calcular vi_t (diferencial logarítmico da volatilidade condicional) para o TGARCH
            if model_choice == "TGARCH":
                vi_t = np.log(vol_tgarch) - np.log(vol_tgarch.shift(1))
                st.write("Diferencial Logarítmico da Volatilidade Condicional (vi_t):")
                st.line_chart(vi_t)

            volatility = calculate_volatility(petr)


            # Plotar gráficos
            #plot_metrics(prices, returns, volatility, ticker_1)

            try:         

                st.subheader(f"Análise Multifractal com {ticker_1}")

                dat1 = np.copy(returns.values)
                dat2 = np.copy(vol_garch.values)

                # Delta 
                trend_base = np.exp(np.cumsum(returns.values)) #index-based

                qorders = list(range(-10, 11))
                #"""
                est_results_xy = basic_dcca(dat1, dat2, Q=qorders, trend_base=trend_base, asymmetry_base='optional')

                # Calculando h_(xy)(q) para os diferentes valores de q
                h_xy_q_neg_10 = (est_results_xy[5] - est_results_xy[8])[0]
                h_xy_q_2 = (est_results_xy[5] - est_results_xy[8])[12]
                h_xy_q_10 = (est_results_xy[5] - est_results_xy[8])[20]

                # Calculando D_(xy)
                D_xy = 0.5 * (np.abs(est_results_xy[2][0] - 0.5) + np.abs(est_results_xy[2][20] - 0.5))

                # Exibindo os resultados de forma organizada e informativa
                st.write(f"Resultado dos cálculos:\n")
                st.write(f"h_(xy)(q = -10)  = {h_xy_q_neg_10:.4f}")
                st.write(f"h_(xy)(q = 2)    = {h_xy_q_2:.4f}")
                st.write(f"h_(xy)(q = 10)   = {h_xy_q_10:.4f}")
                st.write(f"D_(xy)           = {D_xy:.4f}")

                                    # Configurando a base da tendência
                # Renderizar os gráficos
                plot_dcca(dat1, dat2)

                #----------------------------------------

                trend_base = np.exp(np.cumsum(returns[20:].values))
                
                dat_11 = np.copy(returns[20:].values)
                dat_22 = np.copy(volatility[20:].values)  # index-based

                plot_amfdfa_mfadcca(dat_11, dat_22, trend_base)

                #----------------------------------------
                #----------------------------------------

                trend_base = np.exp(np.cumsum(returns[20:].values))
                
                dat_11 = np.copy(returns[20:].values)
                dat_22 = np.copy(volatility[20:].values)  # index-based

                # Chamar a função para exibir o gráfico no Streamlit
                plot_mass_function_tau_q(dat_11, dat_22, trend_base)

                #----------------------------------------
                #----------------------------------------

                trend_base = np.exp(np.cumsum(returns[20:].values))
                
                dat_11 = np.copy(returns[20:].values)
                dat_22 = np.copy(volatility[20:].values)  # index-based

                # Chamar a função para exibir o gráfico
                plot_singularity_spectra(dat_11, dat_22, trend_base)
                #----------------------------------------

                


            except Exception as e:
                st.error(f"Erro na análise MF-ADCCA: {e}")

        with col2:

            st.write(data_2.tail())
            serie_temporal_2 = data_2['Close'].dropna()

            st.title("Análise de Criptomoedas")

            prices2 = serie_temporal_2
            returns2 = calculate_returns(prices2)
            volatility2 = calculate_volatility(returns2)
            # Plotar gráficos
            plot_metrics(prices2, returns2, volatility2, ticker_2)

            try:
                
                st.subheader(f"Análise Multifractal com {ticker_2}")

                dat1_1 = np.copy(returns2[20:].values)
                dat2_2 = np.copy(volatility2[20:].values)

                # Configurando a base da tendência
                # Renderizar os gráficos
                plot_dcca(dat1_1, dat2_2)

                #----------------------------------------


                #----------------------------------------

                
            except Exception as e:
                st.error(f"Erro na análise MF-ADCCA: {e}") 
        
    else:
        st.error("Erro ao carregar dados. Verifique os tickers selecionados ou a conectividade de rede.")
