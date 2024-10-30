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


st.subheader("Rescaled Range Analysis (R/S)")
st.write(" A Method for Detecting Persistence, Randomness, or Mean Reversion in Financial Markets")

ativo = st.selectbox("Escolha um ativo", list(tickers.keys()) + ["Outro (inserir ticker)"])

# Verificar se o usuário escolheu inserir um ticker manualmente
if ativo == "Outro (inserir ticker)":
    ticker = st.text_input("Insira o ticker do ativo")
else:
    ticker = tickers[ativo]

data_inicio = st.date_input("Data de início", datetime(2020, 1, 1))
data_fim = st.date_input("Data de fim", datetime(2024, 1, 1))


if ticker:
    # Adquirir dados históricos do ativo
    data = yf.download(ticker, start=data_inicio, end=data_fim)
    
    if not data.empty:
        st.write(f"Dados carregados para {ativo}")
        st.write(data.tail())

        # Selecionar o preço de fechamento para análise
        serie_temporal = data['Close'].dropna().values

        # Executar a análise R/S
        try:
            st.write("Executando análise R/S...")
            rs_value = rescaled_range_analysis(serie_temporal)

            # Exibir o valor calculado
            st.write(f'Valor R/S calculado: {rs_value:.3f}')

        except Exception as e:
            st.error(f"Erro na análise R/S: {e}")
    else:
        st.error("Erro ao carregar dados. Verifique o ticker selecionado ou a conectividade de rede.")
