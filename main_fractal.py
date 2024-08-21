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



# Funções -------------------------------

# Título do app
st.title("Análise Fractal")

imagem_url = "fractal.jpg"
# Adicione a imagem à barra lateral
st.sidebar.image(imagem_url, caption='Fractalidade', use_column_width=False, output_format="JPEG", width=280)

# Menu na barra lateral
menu = ["Main", "Análise MF-DFA", "Análise R/S", "Comparação de Ativos", "Informações"]
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


elif escolha == "Informações":
    st.subheader("Contato")
    st.write("Para mais informações, entre em contato conosco.")

