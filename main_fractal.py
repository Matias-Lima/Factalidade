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

# Funções -------------------------------

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

plt.style.use('dark_background')


# ----------------------

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


# Funções -------------------------------


# Título do app
st.title("Análise Fractal")

imagem_url = "fractal.jpg"
# Adicione a imagem à barra lateral
st.sidebar.image(imagem_url, caption='Fractalidade', use_column_width=False, output_format="JPEG", width=280)

# Menu na barra lateral
menu = ["Main","Análise MF-DFA", "Análise R/S", "Informações"]
escolha = st.sidebar.selectbox("Navegue", menu)

if escolha == "Main":
    st.subheader("Análise Multifractal")

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


elif escolha == "Informações":
    st.subheader("Contato")
    st.write("Para mais informações, entre em contato conosco.")

