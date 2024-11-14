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

        #График 1: Распределение истинных значений "gt_corners"
        plt.figure()
        sns.histplot(self.data['gt_corners'], kde=True, color='green')
        plt.xlabel('Grand Truth corners')
        plt.title('Distribution of Grand Truth corners')
        plot_path = os.path.join(self.plots_dir, 'gt_corners_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        # График 2: Распределение прогнозируемых значений "rb_corners"
        plt.figure()
        sns.histplot(self.data['rb_corners'], kde=True, color='green')
        plt.xlabel('Predicted corners')
        plt.title('Distribution of Predicted corners')
        plot_path = os.path.join(self.plots_dir, 'rb_corners_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        # График 3: gt_corners и rb_corners
        plt.figure()
        plt.scatter(self.data['gt_corners'], self.data['rb_corners'], color='blue')
        plt.xlabel('Ground Truth Corners')
        plt.ylabel('Model Predicted Corners')
        plt.title('Ground Truth vs Predicted Corners')
        plot_path = os.path.join(self.plots_dir, 'gt_vs_predicted.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        #График 4: Распределение средних отклонений "mean"
        plt.figure()
        sns.histplot(self.data['mean'], kde=True, color='green')
        plt.xlabel('Mean Deviation')
        plt.title('Distribution of Mean Deviation')
        plot_path = os.path.join(self.plots_dir, 'mean_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        #График 5: Распределение максимальных отклонений "max"
        plt.figure()
        sns.histplot(self.data['max'], kde=True, color='red')
        plt.xlabel('Max Deviation')
        plt.title('Distribution of Max Deviation')
        plot_path = os.path.join(self.plots_dir, 'max_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        # График 6: Распределение минимальных отклонений "max"
        plt.figure()
        sns.histplot(self.data['min'], kde=True, color='blue')
        plt.xlabel('Min Deviation')
        plt.title('Distribution of Min Deviation')
        plot_path = os.path.join(self.plots_dir, 'min_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        #График 7: Максимальные отклонения против минимальных отклонений
        plt.figure()
        plt.scatter(self.data['min'], self.data['max'], s=32, alpha=.8)
        plt.xlabel('Min vs Max Distribution')
        plt.title('Distribution of Min-Max Deviation')
        plot_path = os.path.join(self.plots_dir, 'min_max_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        #График 8: Средние отклонения против максимальных отклонений
        plt.figure()
        plt.scatter(self.data['mean'], self.data['max'], s=32, alpha=.8)
        plt.xlabel('Mean vs Max Distribution')
        plt.title('Distribution of Mean-Max Deviation')
        plot_path = os.path.join(self.plots_dir, 'mean_max_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        #График 9: Средние отклонения против минимальных отклонений
        plt.figure()
        plt.scatter(self.data['mean'], self.data['min'], s=32, alpha=.8)
        plt.xlabel('Mean vs Min Distribution')
        plt.title('Distribution of Mean-Min Deviation')
        plot_path = os.path.join(self.plots_dir, 'mean_min_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        #График 10: Распределение средних значений
        plt.figure()
        plt.hist(self.data['mean'], bins=20, alpha=0.5, label='Overall Mean')
        plt.hist(self.data['floor_mean'], bins=20, alpha=0.5, label='Floor Mean')
        plt.hist(self.data['ceiling_mean'], bins=20, alpha=0.5, label='Ceiling Mean')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Mean Values')
        _ = plt.legend()
        plot_path = os.path.join(self.plots_dir, 'mean_Values_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        # График 11: Распределение максимальных значений "max"
        plt.figure()
        plt.hist(self.data['max'], bins=20, alpha=0.5, label='Overall Max')
        plt.hist(self.data['floor_max'], bins=20, alpha=0.5, label='Floor Max')
        plt.hist(self.data['ceiling_max'], bins=20, alpha=0.5, label='Ceiling Max')
        plt.xlabel('Max Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Max Values')
        _ = plt.legend()
        plot_path = os.path.join(self.plots_dir, 'max_Values_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        #График 12: Распределение минимальных значений
        plt.figure()
        plt.hist(self.data['min'], bins=20, alpha=0.5, label='Overall Min')
        plt.hist(self.data['floor_min'], bins=20, alpha=0.5, label='Floor Min')
        plt.hist(self.data['ceiling_min'], bins=20, alpha=0.5, label='Ceiling Min')
        plt.xlabel('Min Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Min Values')
        _ = plt.legend()
        plot_path = os.path.join(self.plots_dir, 'min_Values_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        #plt.show()

        return plot_paths

