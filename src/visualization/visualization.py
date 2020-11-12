from scipy.optimize import curve_fit
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_curve(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y

def learning_curve(data_dict, time, variable,
                   curves=['sigm', 'log', 'inverse'],
                   threshold=0.9, plot=False, zscore=True,
                   x_jitter=None, y_jitter=None, ax=None,
                   plotlabels=True):

    xdata = data_dict[time]
    ydata = data_dict[variable]

    if variable == 'Completion prob':
        ylabel = 'probability'
        zscore = False

    if zscore:
        ydata = stats.zscore(ydata)
        ylabel = 'zscore'
    else:
        ylabel = variable


    # Sigmoid fit
    try:
        p0 = [max(ydata), np.median(xdata),1,min(ydata)]
        popt_sigm, pcov_sigm = curve_fit(sigmoid_curve, xdata, ydata, p0, method='dogbox')
    except:
        pass

    # Distrib-based threshold
    if threshold != None:
        avg = np.mean(ydata[-40:])
        std = np.std(ydata[-40])
        thresh = avg-3*std
        found_thresh = False
        for idx, datum in enumerate(ydata):
            if not found_thresh:
                if datum >= thresh:
                    found_thresh = True
                    idx_thresh = idx
        days_thresh = int(xdata[idx_thresh])
    else:
        days_thresh = 0
        found_thresh = False

    if plot:
        np.random.seed(0)
        if x_jitter != None:
            xdata = np.sort(xdata + np.random.rand(len(xdata))*x_jitter)
        if y_jitter != None:
            ydata = ydata + np.random.rand(len(ydata))*y_jitter
        if ax == None:
            fig, ax = plt.subplots()
        if variable == 'Completion speed' or variable == 'Relative speed':
            ax.invert_yaxis()
        ax.scatter(xdata, ydata, label='data')
        try:
            ax.plot(xdata, sigmoid_curve(xdata, *popt_sigm), 'y-',
                label='sigmoid fit')
        except:
            pass
        if found_thresh:
            ax.axvline(x=xdata[idx_thresh], linestyle='--')
            ax.axhline(y=thresh, linestyle='--')
        ax.legend()
        if plotlabels:
            ax.set(xlabel=time, ylabel=variable)
            ax.xaxis.get_label().set_fontsize(20)
            ax.yaxis.get_label().set_fontsize(20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        return ax, days_thresh
    return days_thresh

def fetch_usable_reps(allvars, data_dict, median_thresh):
    '''
    Fetch filenames of repetitions played after threshold was reached.

    data_dict must be raw, not descriptive metrics
    '''
    usable_filenames = []
    for i, filename in enumerate(allvars['filename']):
        if data_dict['Days of training'][i] >= median_thresh+1:
            usable_filenames.append(allvars['filename'][i][52:])
    return usable_filenames
