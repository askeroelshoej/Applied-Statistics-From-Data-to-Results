
#External functions for the AppStat exam. 
#Author: Aske Roelshøj (KVF851)

#Import:

import numpy as np
import matplotlib.pyplot as plt

from iminuit import Minuit, cost
from iminuit.cost import LeastSquares

import sys
import scipy
from scipy import stats
from sympy import var, solve

from numpy.linalg import inv

from sympy import * 
from sympy import sympify

import pandas as pd


#Nice string outputs from AppStat 23/24 (https://github.com/AppliedStatisticsNBI/AppStat2023/tree/main/External_Functions) : 
def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res

def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))

def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
from iminuit.util import make_func_code
from iminuit import describe #, Minuit,
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2


#Function for reading data using Pandas: 

def data_reader(file):
    data = pd.read_csv(file)
    return data.values

#Histogram binning and plotting function:

def bin_data(data, Nbins, xmin, xmax, xlabel, plot_title = None, plot = False):
    
    counts, bin_edges = np.histogram(data, bins = Nbins, range = (xmin, xmax))
    
    x = 0.5*(bin_edges[:-1] + bin_edges[1:])[counts > 0]
    y = counts[counts > 0]
    sy = np.sqrt(counts[counts > 0])   
    binwidth = bin_edges[1] - bin_edges[0]
        
    if plot and plot_title is not None:
        
        fig, ax = plt.subplots(figsize = (10, 8))
        hist = ax.hist(data, bins = Nbins, range = (xmin, xmax), histtype = 'stepfilled', alpha = 0.25, label = 'Data', color = 'blue')

        ax.errorbar(x, y, yerr = sy, xerr = 0.0, label = 'Data, with Poisson errors', fmt = '.k',  ecolor = 'k', elinewidth = 1, capsize = 1, capthick = 1)

        ax.set(xlabel = xlabel,         
                ylabel = f'Frequency/{binwidth:.{max(0, -int(np.log10(binwidth) - 1 - np.log10(2)))}f}',         
               title = plot_title)   
        ax.legend(loc = 'best', fontsize = 12);       
        
    return x, y, sy, binwidth

#Plotting function: 

def plot_fit(x, y, yerr, fit_func, fit_par, fit_err,names, xlabel, ylabel, parameters = None):
         
    fig, ax = plt.subplots(figsize=(10,8))
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = fit_func(x_fit, *fit_par)

    plt.errorbar(x, y, yerr=yerr, fmt='.k', ecolor='k', elinewidth=1, capsize=1, capthick=1, label = 'Data with errors')
    plt.plot(x_fit, y_fit, '-k', label = r'Fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    d = {}

    Ndecimals = []
    for e in fit_err:
        if e > 0:
            Ndecimals.append(max(0, -int(np.log10(e) - 1 - np.log10(2))))
        else:
            Ndecimals.append(0)

    for i, j in enumerate(names):
        d[f'{j}'] = f'{fit_par[i]:.{Ndecimals[i]}f} ± {fit_err[i]:.{Ndecimals[i]}f}'

    if parameters is not None:
        chi2, ndof, pval = parameters
        d['chi2/ndof'] = f'{chi2:.2f} / {ndof:d}'
        if pval < 1e-10:
            d['p-value'] = f'< 1e-10'
        else:
            d['p-value'] = f'{pval:.4f}'

    text = nice_string_output(d, extra_spacing = 3, decimals = 3)
    add_text_to_ax(0.02, 0.92, text, ax, fontsize = 10, color = 'black')
    ax.text(0.02, 0.97, 'Fit parameters:', weight = 'heavy', fontsize = 12, transform = ax.transAxes, verticalalignment = 'top',  color = 'black')
    plt.legend(fontsize = 12)
    plt.show()

    return

# PLot binned fit: 

def plot_binnedfit(data, Nbins, xmin, xmax, xlabel, fit_func, fit_par, fit_err, names, parameters = None):
        
        x, y, yerr, binwidth = bin_data(data, Nbins, xmin, xmax, xlabel, plot=False)

        fig, ax = plt.subplots(figsize=(10,8))
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = fit_func(x_fit, *fit_par)*binwidth

        plt.errorbar(x, y, yerr=yerr, fmt='.k', ecolor='k', elinewidth=1, capsize=1, capthick=1, label = 'Data with Poisson bars')
        ax.hist(data, bins=Nbins, range=(xmin, xmax), histtype='stepfilled', color='blue', alpha=0.25, label = 'Histogram')
        plt.plot(x_fit, y_fit, '--k', label = 'Fit')
        plt.xlabel(xlabel)
        plt.ylabel(f'Frequency/{binwidth:.2f}')
        
        d = {}

        Ndecimals = []
        for e in fit_err:
            if e > 0:
                Ndecimals.append(max(0, -int(np.log10(e) - 1 - np.log10(2))))
            else:
                Ndecimals.append(0)

        for i, j in enumerate(names):
            d[f'{j}'] = f'{fit_par[i]:.{Ndecimals[i]}f} ± {fit_err[i]:.{Ndecimals[i]}f}'

        if parameters is not None:
            chi2, ndof, pval = parameters
            d['chi2/ndof'] = f'{chi2:.2f} / {ndof:d}'
            d['p-value'] = f'{pval:.4f}'
        

        text = nice_string_output(d, extra_spacing = 3, decimals = 3)
        add_text_to_ax(0.02, 0.92, text, ax, fontsize = 10, color = 'black')
        ax.text(0.02, 0.97, 'Fit parameters:', weight = 'heavy', fontsize = 12, transform = ax.transAxes, verticalalignment = 'top',  color = 'black')
        ax.set_ylim(0, max(y)+0.25*(max(y)-min(y)))
        plt.legend(fontsize = 12)
        plt.show()

        return

#Fitting function using Minuit - not binned data:

def minuit_chi2fit(x, y, yerr, fit_func, fit_guess, names, do_print = False):

    def chi2(x, y, yerr, par):
        y_fit = fit_func(x, *par)
        chi2 = np.sum(((y - y_fit) / yerr)**2)
        return chi2
    
    minimize_object = lambda *par: chi2(x, y, yerr, *par)
    minimize_object.errordef = Minuit.LEAST_SQUARES

    minuit_fit = Minuit(minimize_object, fit_guess)
    minuit_fit.migrad()

    chi2_val = minuit_fit.fval  # Chi2
    ndof = len(x) - len(minuit_fit.values[:]) # Degrees of freedom
    chi2p_val = stats.chi2.sf(chi2_val, ndof) # p-value
    fit_par, fit_err = minuit_fit.values, minuit_fit.errors # Fitted parameters and their errors

    if do_print:
        d = {}

        Ndecimals = []
        for e in fit_err:
            if e > 0:
                Ndecimals.append(max(0, -int(np.log10(e) - 1 - np.log10(2))))
            else:
                Ndecimals.append(0)

        for i, j in enumerate(names):
            d[f'{j}'] = f'{fit_par[i]:.{Ndecimals[i]}f} ± {fit_err[i]:.{Ndecimals[i]}f}'
                
        text = nice_string_output(d, extra_spacing = 3, decimals = 3)
        print('\n' + text)
        print(f'\np(chi2 = {chi2_val:.1f}, ndof = {ndof:d}) = {chi2p_val:6.4f}')
    
    return fit_par, fit_err, chi2_val, ndof, chi2p_val

#Chi2 minuit fit for binned: 

def minuit_binnedchi2fit(data, Nbins, xmin, xmax, fit_func, fit_guess, names, do_print = False):

    x, y, yerr, binwidth = bin_data(data, Nbins, xmin, xmax, xlabel = '', plot=False)

    def chi2(x, y, yerr, par):
        y_fit = fit_func(x, *par)*binwidth
        chi2 = np.sum(((y - y_fit) / yerr)**2)
        return chi2
    
    minimize_object = lambda *par: chi2(x, y, yerr, *par)
    minimize_object.errordef = Minuit.LEAST_SQUARES

    minuit_fit = Minuit(minimize_object, fit_guess)
    minuit_fit.migrad()

    chi2_val = minuit_fit.fval  # Chi2
    ndof = len(x) - len(minuit_fit.values[:]) # Degrees of freedom
    chi2p_val = stats.chi2.sf(chi2_val, ndof) # p-value
    fit_par, fit_err = minuit_fit.values, minuit_fit.errors # Fitted parameters and their errors

    if do_print:
        for i, j in enumerate(names):
            print('Fit', j ,f': {fit_par[i]:.4f} pm {fit_err[i]:.4f}')
        print(f'\np(chi2 = {chi2_val:.1f}, ndof = {ndof:d}) = {chi2p_val:6.4f}')
        
    return fit_par, fit_err, chi2_val, ndof, chi2p_val

# Minuit ULLH fit function: 
def minuit_ullhfit(data, fit_func, fit_guess, names, do_print = False):

    def new_fit_func(x, *par):
        return par[0], fit_func(x, *par) #Our fit function needs N as first parameter. 

    ullhfit = cost.ExtendedUnbinnedNLL(data, new_fit_func)
    minuit_ullh = Minuit(ullhfit, *fit_guess)
    minuit_ullh.migrad();

    fit_par = minuit_ullh.values[:]
    fit_err = minuit_ullh.errors[:]

    if do_print:
        d = {}

        Ndecimals = []
        for e in fit_err:
            if e > 0:
                Ndecimals.append(max(0, -int(np.log10(e) - 1 - np.log10(2))))
            else:
                Ndecimals.append(0)

        for i, j in enumerate(names):
            d[f'{j}'] = f'{fit_par[i]:.{Ndecimals[i]}f} ± {fit_err[i]:.{Ndecimals[i]}f}'
        text = nice_string_output(d, extra_spacing = 3, decimals = 3)
        print('\n' + text)
    return fit_par, fit_err


def hist_with_gauss(data, N_bins, xlabel = None, ax = None):

    # Create a figure if needed
    if ax is None: fig, ax = plt.subplots(figsize = (10,8))

    # Extract values from histogram and outline data
    counts, bin_edges, _ = ax.hist(data, bins = N_bins, alpha = 0.5, histtype='stepfilled', label = 'Histogram', color = 'blue')
    bin_center = (bin_edges[1:] + bin_edges[:-1])/2
    binwidth = bin_edges[1] - bin_edges[0]
    s_counts = np.sqrt(counts) # Poisson errors on the count in each bin
    
    # Removeing any bins without any counts in them:
    x = bin_center[counts > 0]
    y = counts[counts > 0]
    sy = s_counts[counts > 0]

    # Data with errorbars
    ax.errorbar(x, y, yerr = sy, fmt = '.k',  ecolor = 'k', elinewidth = 1, capsize = 1, capthick = 1, label = 'Data with Poisson errors')
    
    # Perform Gaussian fit
    def func_gauss(x, N, mu, sigma) :
        return N * binwidth * stats.norm.pdf(x, mu, sigma)

    #Using minuit to fit the Gaussian:
    fit_par, fit_err, chi2, ndof, pval = minuit_chi2fit(x, y, sy, func_gauss, fit_guess = [max(y), np.mean(data), np.std(data)], names = ['N', 'mu', 'sigma']) 

    N, mu, sigma = fit_par
    N_err, mu_err, sigma_err = fit_err

    # Plot
    xaxis = np.linspace(min(data), max(data), 100)
    y_gauss = func_gauss(xaxis, *fit_par)
    ax.plot(xaxis, y_gauss, linewidth = 1, color = 'k', label = r'Gaussian $\chi^2$ fit')
     
    d = {'N': [N, N_err],
          'mu': mu,
          'sigma': sigma,
          'Ndof': ndof,
          'Chi2': chi2,
          'Prob': pval}

    text = nice_string_output(d, extra_spacing = 3, decimals = 3)
    add_text_to_ax(0.02, 0.92, text, ax, fontsize = 10, color = 'black')
    ax.text(0.02, 0.97, 'Fit parameters:', weight = 'heavy', fontsize = 12, transform = ax.transAxes, verticalalignment = 'top',  color = 'black')

    ax.legend()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f'Frequency/{binwidth:.{max(0, -int(np.log10(binwidth) - 1 - np.log10(2)))}f}', fontsize = 12)

    print(f"Chi2 value: {chi2:.1f}")   
    print(f"Ndof = {ndof:.0f}")
    print(f"Prob(Chi2,Ndof) = {pval}")
    
    return 

def double_gauss_hist(data, N_bins, fit_guess, xlabel = None, sub_pdfs = True):

    #Fit_guess needs to be of the for [frac, mu1, sigma1, mu2, sigma2]

    fig, ax = plt.subplots(figsize = (12,8))

    # Extract values from histogram and outline data 
    counts, bin_edges, _ = ax.hist(data, bins = N_bins, histtype= 'stepfilled', alpha = 0.25, color = 'blue', label = 'Histogram of data')
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    binwidth = bin_edges[1] - bin_edges[0]
    
    # Poisson errors on the count in each bin
    s_counts = np.sqrt(counts)
    
    # We remove any bins, which don't have any counts in them:
    x = bin_centers[counts > 0]
    y = counts[counts > 0]
    sy = s_counts[counts > 0]

    # Plot data with error
    ax.errorbar(x, y, yerr = sy, fmt = '.k',  ecolor = 'k', elinewidth = 1, capsize = 1, capthick = 1, label = 'Data with Poisson errors')
    
    # Define function for double Gauss (normalised)
    def double_gauss(x, N, frac, mu1, sigma1, mu2, sigma2):
        # Notice we introduce a common parameter frac that determines the proportion of the two functions
        return N * binwidth * (frac * stats.norm.pdf(x, mu1, sigma1) + (1-frac)*stats.norm.pdf(x, mu2, sigma2))

    # Perform fit by minimizing chi2
    chi2 = Chi2Regression(double_gauss, x, y, sy)
    minuit_double = Minuit(chi2, N = len(data), frac = fit_guess[0], mu1 = fit_guess[1], sigma1 = fit_guess[2], mu2 = fit_guess[3], sigma2 = fit_guess[4]) 
    minuit_double.migrad(); 
    
    # Extract chi2 values
    chi2 = minuit_double.fval
    ndof = len(x) - len(minuit_double.values)
    pval = stats.chi2.sf(chi2, ndof)

    # Extract values from fit
    fit_par = minuit_double.values
    fit_err = minuit_double.errors

    names = ['N', 'frac', 'mu1', 'sigma1', 'mu2', 'sigma2']
    frac = [fit_par['frac']]
    mean1 = [fit_par['mu1']]
    sigma1 = [fit_par['sigma1']]
    mean2 = [fit_par['mu2']]
    sigma2 = [fit_par['sigma2']]
    
    # Plot Double Gaussian fit
    xaxis = np.linspace(min(data), max(data), 1000)
    y_double = double_gauss(xaxis, *minuit_double.values)
    ax.plot(xaxis, y_double, linewidth = 2, color = 'black', label = ('Double Gaussian fit'))

    # Plot the sub distributions
    if sub_pdfs:
        
        # Define a regular gaussian
        def gauss(x, N, frac, mu, sigma):
            return N * binwidth * frac * stats.norm.pdf(x, mu, sigma)

        ax.plot(xaxis, gauss(xaxis, len(data), frac[0], mean1[0], sigma1[0]), alpha=0.5, linestyle = 'dashed', color = 'k', label = 'Fit 1' )
        ax.plot(xaxis, gauss(xaxis, len(data), 1-frac[0], mean2[0], sigma2[0]), alpha = 0.5, linestyle = 'dashed', color = 'k', label='Fit 2' )

    Ndecimals = []
    for e in fit_err:
        if e > 0:
            Ndecimals.append(max(0, -int(np.log10(e) - 1 - np.log10(2))))
        else:
            Ndecimals.append(0)

    d = {}
    for i, j in enumerate(names):
        d[f'{j}'] = f'{fit_par[j]:.{Ndecimals[i]}f} ± {fit_err[j]:.{Ndecimals[i]}f}'
    
    d['chi2/ndof'] = f'{chi2:.2f} / {ndof:d}'
    d['p-value'] = f'{pval:.4f}'
        

    text = nice_string_output(d, extra_spacing = 3, decimals = 3)
    add_text_to_ax(0.02, 0.92, text, ax, fontsize = 10, color = 'black')
    ax.text(0.02, 0.97, 'Fit parameters:', weight = 'heavy', fontsize = 12, transform = ax.transAxes, verticalalignment = 'top',  color = 'black')
    ax.set_ylim(0, max(y)+0.25*(max(y)-min(y)))
    ax.set_xlabel(xlabel if xlabel else 'x', fontsize = 12)
    ax.set_ylabel(f'Frequency/{binwidth:.{max(0, -int(np.log10(binwidth) - 1 - np.log10(2)))}f}', fontsize = 12) #no. decimals.
    plt.legend(fontsize = 10)

    
    return fit_par, fit_err
    

#Chauvenet filter:

def chauvenet_filter(data):
    N = len(data)
    mean = np.mean(data)
    std = np.std(data)

    criterion = 1.0 / (2*N)

    filtered_data = []
    for x in data:
        deviation = np.abs(x - mean) / std
        prob = stats.norm.sf(deviation) * 2  # Two-tailed probability
        if prob >= criterion:
            filtered_data.append(x)

    return np.array(filtered_data)

#Weighted mean function: 

def weighted_mean(xs, sigs):
    num = np.sum(xs/(sigs**2))
    denom = np.sum(1/(sigs**2))
    w_x = num / denom
    w_sig = np.sqrt(1/denom)

    chi2 = np.sum((xs-w_x)**2/(sigs**2))
    Ndof = len(xs)-1
    prob = stats.chi2.sf(chi2, Ndof)
    
    return w_x, w_sig, chi2, prob, Ndof

#Spearmans correlation: 
def spearman_correlation(x, y):
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)

    D = x_rank - y_rank
    n = len(x)
    rho_s = 1 - (6 * np.sum(D**2)) / (n * (n**2 - 1))

    return rho_s

#Pearsons correlation:
def pearson_correlation(x, y):
    n = len(x)

    cov = np.cov(x, y, ddof=1)[0, 1]

    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)

    rho = cov/(std_x*std_y)

    return rho

#Function for cutting data some amounts of sigma away from mean: 

def data_cutter(data, cut):

    mean = np.mean(data)
    sigma = np.std(data, ddof = 1)

    low, high = (mean - cut*sigma), (mean + cut*sigma)

    accepted = data[(data >= low) & (data <= high)]
    rejected = data[(data < low) | (data > high)]
    
    return accepted, rejected

#Z-test for comparing a value with error to a known constant, with possibility for two-tailed test:

def z_test_constant(mu, sigma, c, twotails = False):

    z = (mu - c) / sigma

    if twotails == True:
        p = 2*stats.norm.sf(np.abs(z))
    
    if twotails == False:
        p = stats.norm.sf(np.abs(z))

    return z, p

#Z-test for comparing two values with errors, with possibility for two-tailed test:
def z_test_means(mu1, sig1, mu2, sig2, twotails = False):
    z = (mu1 - mu2) / np.sqrt(sig1**2 + sig2**2)

    if twotails == False:
        p = stats.norm.sf(abs(z))  # one-sided p-value
    if twotails == True:
        p = 2*stats.norm.sf(abs(z))  # two-sided p-value
    
    return z, p

def ks_test_2samp(data1, data2):
    ks_stat, p_value = stats.ks_2samp(data1, data2)
    return ks_stat, p_value

# Seperation calculator for Fisher's discriminant analysis:
def calc_separation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    d = np.abs((mean_x - mean_y)) / np.sqrt(std_x**2 + std_y**2)
    
    return d

def fisher_LDA(X, y, nspec, nvar, Nbins, spec_name1, spec_name2):
    
    covariances = np.zeros((nvar, nvar, nspec))

    for ivar in range(nvar):
        for jvar in range(nvar):
            for ispec in range(nspec):
                data_ispec = X[y == ispec]
                data_spec_ivar = data_ispec[:, ivar]
                data_spec_jvar = data_ispec[:, jvar]
                covariances[ivar, jvar, ispec] = np.cov(data_spec_ivar, data_spec_jvar, ddof = 1)[0,1]

    covmat_spec1 = covariances[:, :, 0]
    covmat_spec2 = covariances[:, :, 1]

    covmat_comb = covmat_spec1 + covmat_spec2
    print('Combined covariance matrix:\n', covmat_comb)
    print('')
    covmat_comb_inv = inv(covmat_comb)
    print('Inverse of combined covariance matrix:\n', covmat_comb_inv)
    

    mu = np.zeros((nspec, nvar))
    for ispec in range(nspec):
        data_ispec = X[y == ispec]
        for ivar in range(nvar):
            data_spec_ivar = data_ispec[:, ivar]
            mu[ispec, ivar] = np.mean(data_spec_ivar)
    
    wf = covmat_comb_inv @ (mu[1] - mu[0])
    print('Fisher discriminant weights:\n', wf)
    
    F_spec1 = X[y == 0] @ wf
    F_spec2 = X[y == 1] @ wf

    d = calc_separation(F_spec1, F_spec2)
    
    fig, ax = plt.subplots(figsize = (8,6))
    ax.hist(F_spec1, bins = Nbins, label = f"{spec_name1}", histtype = "stepfilled", alpha = 0.5, color = 'blue')
    ax.hist(F_spec2, bins = Nbins, label = f"{spec_name2}", histtype = "stepfilled", alpha = 0.5, color = 'red')
    bin_width = (max(np.concatenate((F_spec1, F_spec2))) - min(np.concatenate((F_spec1, F_spec2)))) / Nbins
    plt.title(rf"Distribution of $\mathcal{{F}}$-values for {spec_name1} and {spec_name2}", fontsize = 12)

    plt.xlabel(r"$\mathcal{F}$-values", fontsize = 12)
    plt.ylabel(f"Frequency/{bin_width:.2f}", fontsize = 12)

    d_text = {r'$\Delta$': f'{d:.2f}'}
    text = nice_string_output(d_text, extra_spacing = 3, decimals = 3)
    add_text_to_ax(0.02, 0.92, text, ax, fontsize = 12, color = 'black')
    ax.text(0.02, 0.97, 'Separation', weight = 'heavy', fontsize = 12, transform = ax.transAxes, verticalalignment = 'top',  color = 'black')

    plt.legend()
    
    return F_spec1, F_spec2, wf, d



