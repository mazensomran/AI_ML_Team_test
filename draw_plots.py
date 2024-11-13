import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class PlotDrawer:
    def __init__(self, data):
        self.data = pd.read_json(data)
        self.plots_dir = "plots"


    def draw_plots(self):

        os.makedirs(self.plots_dir, exist_ok=True)
        plot_paths = []

        #График 1: gt_corners и rb_corners
        #plt.figure(num=1)
        plt.plot(self.data['gt_corners'], self.data['rb_corners'], color='blue')
        plt.xlabel('Ground Truth Corners')
        plt.ylabel('Model Predicted Corners')
        plt.title('Ground Truth vs Predicted Corners')
        plot_path = os.path.join(self.plots_dir, 'gt_vs_predicted.png')
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.show()

        # График 2: Распределение асимметрии «mean»
        #plt.figure(num=2)
        sns.histplot(self.data['mean'], kde=True, color='green')
        plt.xlabel('Mean Deviation')
        plt.title('Distribution of Mean Deviation')
        plot_path = os.path.join(self.plots_dir, 'mean_distribution.png')
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.show()

        # График 3: Отношение отклонений между столбцами: Mean, Min, Max.
        #plt.figure(num=3)
        plt.plot(self.data['mean'], label='Mean')
        plt.plot(self.data['min'], label='Min')
        plt.plot(self.data['max'], label='Max')
        plt.xlabel('Data Points')
        plt.ylabel('Deviation in Degrees')
        plt.legend()
        plt.title('Mean, Min, and Max Deviations')
        plot_path = os.path.join(self.plots_dir, 'deviation_comparison.png')
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.show()

        return plot_paths

