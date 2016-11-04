
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

my_locator = MaxNLocator(6)

#legends = {1:'Original SDX', 2:'CP Isolation', 3:'NH Encoding', 0:'Cardigan', 4: 'SuperSet'}
#legends = {1:'SDX', 0:'Baseline', 2:'Distributed CP', 3:'NH Encoding',4: 'X-SDX', 5:'Optimal'}
#legends = {1:'C-SDX', 2:'D-SDX', 3:'ND-SDX', 0:'Vanilla-SDX', 4: 'RND-SDX', 5:'Baseline'}
#legends = {1:'MDS SDX-Central', 2:'iSDX-D', 3:'iSDX-N', 0:'Unoptimized', 4: 'iSDX-R', 5:'Optimal'}
hatches = {0:'/', 1:'-', 2:'+', 3:'o', 4:'\\', 5:'x', 6:'o', 7:'O', 8:'.'}

color_n={0:'g',1:'m',2:'c',3:'r',4:'b', 5:'r'}
markers={0:'o',1:'*',2:'^',3:'s',4:'d', 5:'s'}


def mboxPlot(dname, params):

    data = json.load(open(dname, 'r'))

    data["plot"] = {}
    xlab=[]
    for param in params:
        print param
        data["plot"][param] = {}
        modes = [int(x) for x in data[param].keys()]
        modes.sort()
        for mode in modes:
            v = data[param][unicode(mode)]
            data["plot"][param][mode] = {}
            tmp_median={}
            tmp_stddev={}
            #print mode, v
            xlab = [int(x) for x in v.keys()]
            xlab.sort()
            #print xlab

            for part in xlab:
                v2 = v[unicode(part)]
                # TODO: Avoid hardcoding...
                for ind in range(8):
                    #if ind == 3 and mode ==3:
                        #print part, v2
                    if ind+1 not in tmp_median:
                        tmp_median[ind+1] = []
                        tmp_stddev[ind+1] = []
                    tmp_median[ind+1].append(numpy.median([x[ind] for x in v2]))
                    tmp_stddev[ind+1].append(numpy.std([x[ind] for x in v2]))

            data["plot"][param][mode]['median'] = tmp_median
            data["plot"][param][mode]['stddev'] = tmp_stddev

        # Logic to add the data for mode 5
        mode = 4
        mode_new = 5
        data["plot"][param][mode_new] = {}

        data["plot"][param][mode_new]['median'] = {}
        data["plot"][param][mode_new]['stddev'] = {}

        data["plot"][param][mode_new]['median'] = {6:data["plot"][param][mode]['median'][7]}
        data["plot"][param][mode_new]['stddev'] = {6:data["plot"][param][mode]['stddev'][7]}

    #print data["plot"]

    return data

def plotControlPlaneCompute(dname, params):
    param = params[0]
    with open(dname, 'r') as f:
        data = json.load(f)
        compilation_time = {}
        for mode in data[param]:
            compilation_time[mode] = {}
            tmp_median = []
            tmp_stddev = []
            v = data[param][unicode(mode)]
            xlab = [int(x) for x in v.keys()]
            xlab.sort()
            print xlab

            for part in xlab:
                # Compilation time for  1000000  rules is:  66.7696409225
                #print "Mode: ", mode, " part: ", part
                if int(mode) == 0:
                    # No MDS time for this scheme
                    print data[param][mode][unicode(part)]
                    tmp = [(float(x[5]*66.7696409225)/1000000) for x in data[param][mode][unicode(part)]]
                else:
                    # MDS time needs to be taken into consideration
                    tmp1 = [x[2] for x in data[param][mode][unicode(part)]]
                    tmp2 = [(float(x[5]*66.7696409225)/1000000) for x in data[param][mode][unicode(part)]]
                    #print "tmp1: ", tmp1
                    #print "tmp2: ", tmp2
                    tmp = map(add, tmp1, tmp2)
                #print "tmp: ", tmp

                tmp_median.append(numpy.median(tmp))
                tmp_stddev.append(numpy.std(tmp))
            compilation_time[mode]["median"] = tmp_median
            compilation_time[mode]['stddev'] = tmp_stddev

        print "compilation time: ", compilation_time
        data["plot"] = compilation_time
        # Plot this graph
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        p1 = []
        i = 0

        # Motivation Case
        legnd=[]
        leg = [0, 1, 4]

        for mode in leg:
            #leg.append(float(k))
            #print data[param].keys()
            v = data[param][unicode(mode)]
            #print v
            xlab = [int(x) for x in v.keys()]
            xlab.sort()

            a = data["plot"][unicode(mode)]['median']
            err = data["plot"][unicode(mode)]["stddev"]
            print mode, a

            pl.errorbar(xlab, a, yerr=err, markerfacecolor=color_n[mode],
                          color='k', ecolor='k', marker=markers[mode], label= legends[mode])
            #print p1[i]
            i+=1


        ax.yaxis.set_major_locator(my_locator)
        #pl.yscale('log')
        ax.set_ylim(ymin = -200)
        ax.set_xlim(xmin = 10)
        #pl.xlim(-2,61)

        leg.sort()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, fancybox=True, shadow=False)
        pl.xlabel('Participants')
        pl.ylabel('Time (s)')

        ax.grid(True)
        plt.tight_layout()
        plot_name = param+'_motivations_cpTime.eps'
        pl.savefig(plot_name)


def parse_data(fname):
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

def plot_data(data):
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
    plot_name = "service_chaining_minbits.pdf"
    pl.savefig(plot_name)
    pl.show()


if __name__ == '__main__':
    fname = 'ordered_widths_experiment_40_800.json'
    data = parse_data(fname)
    plot_data(data)
