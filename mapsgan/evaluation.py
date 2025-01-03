from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
from mapsgan.utils import smooth_data
from matplotlib.pyplot import cm


class Evaluation:
    """This class contains evaluation metrics."""
    NotImplemented

class PlotProps:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    def __init__(self):
        pass

    # Set figure and subplot properties

    def init_figure(self, figsize, hspace=.3, wspace=.1):
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        return fig

    def init_subplot(self, title,
                     tot_tup=(1, 1), sp_tup=(0, 0),
                     colspan=1, rowspan=1,
                     sharex=None, sharey=None,
                     xlabel='', ylabel='',
                     despine=True,
                     offset=5, trim=False,
                     ttl_fs=15, ttl_pos='center'):

        ax = plt.subplot2grid(tot_tup, sp_tup, colspan, rowspan, sharex=sharex, sharey=sharey)
        ax.set_title(title, fontsize=ttl_fs, loc=ttl_pos)

        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)

        sns.set(context='paper', style='ticks', font_scale=1.5)
        sns.axes_style({'axes.edgecolor': '.6', 'axes.linewidth': 5.0})
        if despine is True:
            sns.despine(ax=ax, offset=offset, trim=trim)

        return ax

    def legend(self, loc='best', fontsize=15):
        plt.legend(loc=loc, fontsize=fontsize, frameon=False)


class Visualization(Evaluation):
    """Create plots. Can leverage the evaluation metrics."""
    plot = PlotProps()

    def __init__(self):
        pass

    def distribution(self, predictions, legend=False, ground_truth=False, input_truth=False):
        """

        Args:
            predictions (dict): a disctionary of lists (output of Solver.sample_distribution)
            legend (bool):
            ground_truth (bool):
            input_truth (bool):

        Returns:

        """
        pass

    def trajectories(self, output, scenes = [2], legend=False, ground_truth=False, input_truth=False, dist=False, xlim=None, ylim=None):
        """

        Args:
            output: a dictionary that contains the output of Solver.test
            scenes: a list of integer(s).
                    if it is a single integer then that many random scenes are shown,
                    if list of integers then the scenes that were indicated by the list are shown

        Returns: subplots of random or selected scenes containing
                 input trajectories as filled large dots
                 predicted trajectories as x
                 ground truth as small dots

        """
        if isinstance(scenes, list):
            scenes_list = scenes
            num_scenes = len(scenes)+1
        elif not scenes: #None: plot all
            num_scenes = len(output['xy_in'])
            scenes_list = list(range(num_scenes))
        else: #int: plot int number of random scenes
            num_scenes = scenes
            scenes_list = np.random.randint(len(output['xy_in']), size=num_scenes)

        gridwidth = int(np.ceil(np.sqrt(num_scenes)))+1
        gridheight = gridwidth if gridwidth * (gridwidth - 1) < num_scenes else (gridwidth -1)-1
        figsize = [5*gridwidth, 5*gridheight]

        ymin = np.min([np.min(seq[:, :, 1]) for scene in output.values() for seq in scene]) - 0.1
        ymax = np.max([np.max(seq[:, :, 1]) for scene in output.values() for seq in scene]) + 0.1
        xmin = np.min([np.min(seq[:, :, 0]) for scene in output.values() for seq in scene]) - 0.1
        xmax = np.max([np.max(seq[:, :, 0]) for scene in output.values() for seq in scene]) + 0.1

        # sns.set_context('poster')
        fig = self.plot.init_figure(figsize)
        max_a = 0
        i=0
        ax = self.plot.init_subplot(type, tot_tup=(gridheight, gridwidth),
                                    sp_tup=(int(i // gridwidth), int(i % gridwidth)))

        for i, s in enumerate(scenes_list):
            num_agents = output['xy_in'][s].shape[1]
            max_a = max(max_a,num_agents)

            color = ['b', 'orange', 'g', 'r', 'purple', 'k']
            if dist:
                ax = self.plot.init_subplot(type, tot_tup=(gridheight, gridwidth), sp_tup=(int(i // gridwidth), int(i % gridwidth)))
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            #ax.set_xticks([])
            #ax.set_yticks([])

            for a in range(num_agents):
                if input_truth:
                    ax.plot( output['xy_in'][s][:, a, 0], output['xy_in'][s][:, a, 1],
                             'o-', c=color[a%len(color)], markersize=5, label=f'Input Agent {a}')
                if ground_truth:
                    ax.plot( output['xy_out'][s][:, a, 0], output['xy_out'][s][:, a, 1],
                             '.-', c=color[a%len(color)], markersize=5, label=f'Output Agent {a}')
                ax.plot( output['xy_pred'][s][:, a, 0], output['xy_pred'][s][:, a, 1],
                         'x-', c=color[a%len(color)], markersize=5,  label=f'Prediction Agent {a}')
            ax.set_title(s)

        if legend:
            lines = []
            for a in range(max_a):
                lines.append(mlines.Line2D([], [], color=color[a%len(color)], marker='o', c=color[a%len(color)],
                                          markersize=10, label=f'Input Agent {a+1}'))
                if ground_truth:
                    lines.append(mlines.Line2D([], [], color=color[a%len(color)], marker='.', c=color[a%len(color)],
                                          markersize=10, label=f'Groundtruth Agent {a+1}'))
                lines.append(mlines.Line2D([], [], color=color[a%len(color)], marker='x', c=color[a%len(color)],
                                      markersize=10, label=f'Prediction Agent {a+1}'))

            fig.legend(handles=lines, loc=(0.6, 0.055))
            #plt.show()

    def loss(self, loss_history, types = None, figsize = [16, 4], figtitle = ''):
        """Plot losses.

        Args:
            loss_history (dict): Loss history attribute of a solver.
            types (list of str): Specify which losses to plot.

        Returns:
            ax object
        """
        losses = loss_history['generator']
        losses.update(loss_history['discriminator'])
        keyswitch = {'D_real': 'D_Real', 'D_fake': 'D_Fake', 'G_gan': 'G_BCE', 'G_norm': 'G_L1',
                     'G_BCE': 'G_BCE', 'G_L1': 'G_L1', 'G_KL': 'G_KL', 'D_Real':'D_Real', 'D_Fake':'D_Fake',
                     'G_L1z':'G_L1z'}
        if not types:
            types = losses.keys()
        losses = {keyswitch[type]:loss for type, loss in losses.items() if type in types}  # filter out types
        num_axes = len(losses)
        fig = self.plot.init_figure(figsize)
        for i, (type, loss) in enumerate(losses.items()):
            ax = self.plot.init_subplot(keyswitch[type], tot_tup=(1, num_axes), sp_tup=(0, i))
            ax.plot(loss)
            ax.set_xlabel('Checkpoints')
            #ax.set_yticks([])
            if i == 0:
                ax.set_ylabel('Loss (a.u.)')
        if figtitle:
            fig.suptitle(figtitle)

    def loss_val(self, loss_history, smoothing=5, figsize = [16,4]):
        losses = loss_history
        keys = losses.keys()

        plt.figure(figsize=figsize)
        for i,k in enumerate(keys):
            if losses[k]:
                plt.subplot(1,len(keys),i+1)
                plt.plot(smooth_data(losses[k], smoothing))
                plt.title(k)
        plt.show()

