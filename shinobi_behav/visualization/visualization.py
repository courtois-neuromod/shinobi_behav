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
    if 'sigm' in curves:
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
        if 'sigm' in curves:
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

def lcurves_plot(data_dict, variables, title):
    fig, axes = plt.subplots(len(variables), 1, figsize=(5,20))

    for idx, var in enumerate(variables):
        ax, idx_thresh = learning_curve(data_dict, 'Days of training', var,
                                    zscore=False, plot=True,
                                    x_jitter=1, y_jitter=0.1,
                                    ax=axes[idx], plotlabels=True, threshold = None, curves=[])
    # Adjust overall plots
    for ax in fig.get_axes():
        ax.label_outer()
    fig.suptitle(title, y=0.92, fontsize=20)
    return fig

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

def plot_bidsevents(events_df):
    event_ends = []
    for i in range(len(merged_df['onset'])):
        event_ends.append(merged_df['onset'][i] + merged_df['duration'][i])

    total_duration = max(event_ends)
    time_axis = np.linspace(0, total_duration, 10000)


    dict_to_plot = {}
    for ev_type in merged_df['trial_type'].unique():
        dict_to_plot[ev_type] = np.zeros(len(time_axis))

    for idx, line in merged_df.iterrows():
        for i, timepoint in enumerate(time_axis):
            if timepoint >= line['onset'] and timepoint <= line['onset']+line['duration']:
                dict_to_plot[line['trial_type']][i] = 1
    fig = plt.plot()
    for i, key in enumerate(dict_to_plot.keys()):
        plt.scatter(time_axis, dict_to_plot[key]*(len(dict_to_plot.keys())-i), label=key)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    return fig
