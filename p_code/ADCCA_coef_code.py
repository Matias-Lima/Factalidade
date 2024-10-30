#################################################
# codes for implementing DFA-based analysis
# S. Kakinaka and K. Umeno, Exploring asymmetric multifractal cross-correlations of price-volatility and asymmetric volatility dynamics in cryptocurrency markets. Physica A 581, 126237 (2021) https://doi.org/10.1016/j.physa.2021.126237
# Shinji Kakinaka: kakinaka.shinji.35e@st.kyoto-u.ac.jp
#################################################
import numpy as np
import math
###########################################
"""
references:
    [1] G. Cao, L.-Y. He, J. Cao, et al., Multifractal Detrended Analysis Method and its Application in Financial Markets, Springer, 2018. (A-DCCA coefficient)
    [2] G.F. Zebende, DCCA cross-correlation coefficient: Quantifying level of cross-correlation, Physica A 390 (4) (2011) 614–618. (DCCA coefficient)
    [3] B. Podobnik, Z.-Q. Jiang, W.-X. Zhou, H.E. Stanley, Statistical tests for power-law cross-correlated processes, Phys. Rev. E 84 (6) (2011) 066118. (DCCA coefficient)
*** note that this method CANNOT be directly extended to multifractal cases ***
"""
def dccacoef(x, y, S, m, skip_agg=False, trend_base=None, asymmetry_base='return'):
    N = len(x)
    assert len(y) == N, '{} segments'.format(len(y))
    if skip_agg:
        X = np.copy(x)
        Y = np.copy(y)
    else:
        X = np.cumsum(x)
        Y = np.cumsum(y)

    def f2dcca(v, s, correlation):
        """
            (covariance of the residuals)
            f^2_dcca(s, v)
        """
        Ns = N-s+1
        ax = np.arange(1, s+1)
        segment_x = X[v-1:v-1+s]
        segment_y = Y[v-1:v-1+s]
        coef_x = np.polyfit(ax, segment_x, m)
        coef_y = np.polyfit(ax, segment_y, m)
        fitting_x = np.polyval(coef_x, ax)
        fitting_y = np.polyval(coef_y, ax)
        if correlation == 'DCCA':
            return np.mean((segment_x - fitting_x)*(segment_y - fitting_y)) # DCCA for x and y
        elif correlation == 'DFAX':
            return np.mean((segment_x - fitting_x)**2) # DFA for x
        elif correlation == 'DFAY':
            return np.mean((segment_y - fitting_y)**2) # DFA for y
        else:
            print('not available: DCCA or DFAX or DFAY only\nPlease try again')
    
    if asymmetry_base == 'index':
        """ Which data to use for dividing the trends """
        x_alt = np.exp(np.cumsum(x)) # index-based (level data) !
    elif asymmetry_base == 'return':
        x_alt = np.copy(x) # return-based
    elif asymmetry_base == 'optional':
        x_alt = np.copy(trend_base) # or either optional trend criterion
    else:
        print('not available: index or return or optional only\nPlease try again')

    def asym_trend(v, s):
        """
        Detecting trend in some data series H
        L_H(x)=a_H+b_H x
        """
        Ns = N-s+1
        ax = np.arange(1, s+1)
        segment = x_alt[v-1:v-1+s]
        coef = np.polyfit(ax, segment, 1) # m=1 for linear function y=ax+b (a,b)
        return coef[0] #a

    Q = [2] # order parameter q=2
    for i, q in enumerate(Q):
        Fqs_dcca = np.zeros(len(S))
        Fqs_dcca_plus = np.zeros(len(S))
        Fqs_dcca_minus = np.zeros(len(S))
        Fqs_dfax = np.zeros(len(S))
        Fqs_dfax_plus = np.zeros(len(S))
        Fqs_dfax_minus = np.zeros(len(S))
        Fqs_dfay = np.zeros(len(S))
        Fqs_dfay_plus = np.zeros(len(S))
        Fqs_dfay_minus = np.zeros(len(S))

        for j, s in enumerate(S):
            Ns = N-s+1
            segs = np.array([f2dcca(v, s, correlation='DCCA') for v in range(1, Ns + 1)])
            segs_dfax = np.array([f2dcca(v, s, correlation='DFAX') for v in range(1, Ns + 1)])
            segs_dfay = np.array([f2dcca(v, s, correlation='DFAY') for v in range(1, Ns + 1)])

            assert len(segs) == Ns, '{} segments'.format(len(segs))
            assert len(segs_dfax) == Ns, '{} segments'.format(len(segs_dfax))
            assert len(segs_dfay) == Ns, '{} segments'.format(len(segs_dfay))
            
            #trend_segs = np.array([
            #    [asym_trend(v, s) for v in range(1, Ns + 1)],
            #    [asym_trend(v, s, reverse=True) for v in range(Ns+1, 2 * Ns + 1)]
            #]).reshape(-1) # for non-overlapping strategy
            trend_segs = np.array([asym_trend(v, s) for v in range(1, Ns + 1)])
            
            assert len(trend_segs) == Ns, '{} segments'.format(len(trend_segs))
            
            M_plus = np.sum((1+np.sign(trend_segs))/2)
            M_minus = np.sum((1-np.sign(trend_segs))/2)
            
            assert M_plus + M_minus == Ns, '{} segments for M+ + M-'.format(M_plus+M_minus)
            
            # calculate qth fluctuation function
            # asymmetric version
            Fqs_dcca_plus[j] = (np.sum(((1+np.sign(trend_segs))/2)*segs) /M_plus)
            Fqs_dcca_minus[j] = (np.sum(((1-np.sign(trend_segs))/2)*segs) /M_minus)
            Fqs_dfax_plus[j] = (np.sum(((1+np.sign(trend_segs))/2)*segs_dfax) /M_plus)
            Fqs_dfax_minus[j] = (np.sum(((1-np.sign(trend_segs))/2)*segs_dfax) /M_minus)
            Fqs_dfay_plus[j] = (np.sum(((1+np.sign(trend_segs))/2)*segs_dfay) /M_plus)
            Fqs_dfay_minus[j] = (np.sum(((1-np.sign(trend_segs))/2)*segs_dfay) /M_minus)
            # overall version
            Fqs_dcca[j] = np.mean(segs)
            Fqs_dfax[j] = np.mean(segs_dfax)
            Fqs_dfay[j] = np.mean(segs_dfay)
                
        # DCCA cross-correlation coefficient
        # asymmetric version
        CCC_plus = Fqs_dcca_plus/np.sqrt(Fqs_dfax_plus*Fqs_dfay_plus)
        CCC_minus = Fqs_dcca_minus/np.sqrt(Fqs_dfax_minus*Fqs_dfay_minus)
        # overall version
        CCC = Fqs_dcca/np.sqrt(Fqs_dfax*Fqs_dfay)
        
    # 0: scale,1: overall, 2: positive, 3: negative
    return S, CCC, CCC_plus, CCC_minus

def basic_dccacoef(x, y, m=2, skip_agg=False, observations=100, trend_base=None, asymmetry_base='return'):
    N = len(x)
    """ recommended;
    #s_min = max(20, int(np.floor(N/100)))
    #s_max = min(20*s_min, int(np.floor(N/10)))
    """
    s_min = 10
    s_max = N/5
    s_inc = (s_max - s_min) / (observations-1)
    S = [s_min + int(np.floor(i*s_inc)) for i in range(0, observations)] # change the number of observations to make computational time shorter
    return dccacoef(x, y, S=S, m=m, skip_agg=skip_agg, trend_base=trend_base, asymmetry_base=asymmetry_base)
