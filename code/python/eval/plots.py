
import json
import matplotlib
import numpy as np
matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import pylab as pl
#pl.switch_backend('PS')
#matplotlib.use('PS')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from plotlib import *

my_locator = MaxNLocator(6)

#legends = {1:'Original SDX', 2:'CP Isolation', 3:'NH Encoding', 0:'Cardigan', 4: 'SuperSet'}
#legends = {1:'SDX', 0:'Baseline', 2:'Distributed CP', 3:'NH Encoding',4: 'X-SDX', 5:'Optimal'}
#legends = {1:'C-SDX', 2:'D-SDX', 3:'ND-SDX', 0:'Vanilla-SDX', 4: 'RND-SDX', 5:'Baseline'}
#legends = {1:'MDS SDX-Central', 2:'iSDX-D', 3:'iSDX-N', 0:'Unoptimized', 4: 'iSDX-R', 5:'Optimal'}
hatches = {0:'/', 1:'-', 2:'+', 3:'o', 4:'\\', 5:'x', 6:'o', 7:'O', 8:'.'}

color_n={0:'g',1:'m',2:'c',3:'r',4:'b', 5:'r'}
markers={0:'o',1:'*',2:'^',3:'s',4:'d', 5:'s'}

def parse_data_mb(fname):
    data = {}
    with open(fname, 'r') as f:
        dump = json.load(f)
        for mb in dump:
            mbn = int(str(mb)[-2:])
            data[mbn] = {}
            for pt in dump[mb]:
                ptn = int(str(pt)[-3:])
                data[mbn][ptn] = dump[mb][pt]

    return data

def plot_data_mb(data, plot_name):
    legends = data.keys()
    legends.sort()
    x_labs = data[legends[0]].keys()
    x_labs.sort()
    print legends
    print x_labs
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ctr = 0
    for leg in legends:
        y = [np.median(data[leg][x]) for x in x_labs]
        yerr = [np.std(data[leg][x]) for x in x_labs]
        print leg, y, yerr
        if str(leg) == '40':
            leg = str('40 Middle Boxes')
        pl.errorbar(x_labs, y, yerr=yerr, markerfacecolor=color_n[ctr],
                      color='k', ecolor='k', marker=markers[ctr], label= str(leg))
        ctr += 1
        #break
    ax.yaxis.set_major_locator(my_locator)
    #pl.yscale('log')
    ax.set_ylim(ymin = 8)
    #ax.set_ylim(ymax = 24)
    ax.set_xlim(xmin = 90)
    ax.set_xlim(xmax = 810)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,
                fancybox=True, shadow=False)
    pl.xlabel('Number of Unique Paths')
    pl.ylabel('Minimum Bits Required')
    ax.grid(True)
    plt.tight_layout()
    pl.savefig(plot_name)
    pl.show()


def parse_data_cdf():
    flat_fname = "o_flattag_mem_dict.json"
    flex_fname = "varyingwidths_40_800_dict.json"
    data = {}
    with open(flat_fname, 'r') as f:
        dump = json.load(f)
        data[0] = []
        for attr in dump[u'flat']:
            data[0] += dump[u'flat'][attr]

    with open(flex_fname, 'r') as f:
        dump = json.load(f)
        for mbits in dump:
            if "flat" not in mbits:
                mbits_n = int(str(mbits)[-2:])
                print mbits_n
                data[mbits_n] = []
                for attr in dump[mbits]:
                    data[mbits_n] += dump[mbits][attr]
    #print data
    order = data.keys()
    order.sort()
    plotCDF(data, order, 'Number of Forwarding Table Entries', 'CDF of Attributes', 'N/A', 'N/A', "attr_cdf")


if __name__ == '__main__':
    fname = 'ordered_widths_experiment_40_800.json'
    #fname = 'unordered_widths_experiment.json'
    plot_name = "service_chaining_minbits_ordered.eps"
    #plot_name = "service_chaining_minbits_unordered.eps"
    data = parse_data_mb(fname)
    plot_data_mb(data, plot_name)

    #plot_data_cdf()
