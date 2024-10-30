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
#st.set_page_config(layout="wide")

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📈",
)


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


# Funções Acima -------------------------------

# ---------------------- Dicionários

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

# -------------------------------

# Título do app
st.title("Análise Fractal")

imagem_url = "fractal.jpg"
# Adicione a imagem à barra lateral
st.sidebar.image(imagem_url, caption='Fractalidade', use_column_width=False, output_format="JPEG", width=280)

# Menu na barra lateral
menu = ["Main", "Análise MF-DFA", "Análise R/S", "Comparação de Ativos","Análise MF-ADCCA" ,"Informações"]
escolha = st.sidebar.radio("Navegue", menu)
st.sidebar.divider()


if escolha == "Main":
    st.subheader("Análise Multifractal")

    st.write("Navegue pelo menu, basta clicar no > sidebar no lado superior esquerdo")

    st.write("The study of financial or crude oil markets is largely based on current main stream literature, whose fundamental assumption is that stock price (or returns) follows a normal distribution and price behavior obeys ‘random-walk’ hypothesis (RWH), which was first introduced by Bachelier (1900), since then it has been adopted as the essence of many asset pricing models. However, some important results in econophysics suggest that price (or returns) in financial or commodity markets have fundamentally different properties that contradict or reject RWH. These ubiquitous properties identified are: fat tails (Gopikrishnan 2001), long-term correlation (Alvarez 2008), volatility clustering (Kim 2008), fractals multifractals (He et al. 2007), chaos (Adrangi 2001), etc. Nowadays, RWH has been widely criticized in the finance and econophysics literature as this hypothesis fails to explain the market phenomena.")

    st.write("Fractal methods are divided into single fractal and multifractal methods. Single-fractal analysis is mainly the long memory (long memory) (also known as Persistence) or anti-persistence. The long-term memory (long-range correlation) in the financial time series is mainly judged by the Hurst index estimated by various methods")

    st.write(" Evidence of H differences from a half (1/2) could be interpreted as proof that returns are not independent and that long-term memory is present (Peters 1994, 1996). This shows that the volatility of securities prices to a certain extent, there is predictability. Rachev (2010), Paolella (2016), Francq (2016) etc. have confirmed that the fractal distribution have freat adaptability in the financial market, and opened up a new path for the financial market forecast.")


elif escolha == "Análise MF-DFA":
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
                min_value = st.number_input("Valor mínimo", min_value=-100, max_value=100, value=-60, step=1)
                max_value = st.number_input("Valor máximo", min_value=-100, max_value=100, value=60, step=1)
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


elif escolha == "Comparação de Ativos":

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


elif escolha == "Análise R/S":
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


elif escolha == "Análise MF-ADCCA":

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

                    #plot_amfdfa_mfadcca(dat_11, dat_22, trend_base)

                    #----------------------------------------
                    #----------------------------------------

                    trend_base = np.exp(np.cumsum(returns[20:].values))
                    
                    dat_11 = np.copy(returns[20:].values)
                    dat_22 = np.copy(volatility[20:].values)  # index-based

                    # Chamar a função para exibir o gráfico no Streamlit
                    #plot_mass_function_tau_q(dat_11, dat_22, trend_base)

                    #----------------------------------------
                    #----------------------------------------

                    trend_base = np.exp(np.cumsum(returns[20:].values))
                    
                    dat_11 = np.copy(returns[20:].values)
                    dat_22 = np.copy(volatility[20:].values)  # index-based

                    # Chamar a função para exibir o gráfico
                    #plot_singularity_spectra(dat_11, dat_22, trend_base)
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


elif escolha == "Informações":
    st.subheader("Contato")
    st.write("Para mais informações, entre em contato conosco.")

