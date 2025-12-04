# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class ModelVisualizer:
    def __init__(self, layer_info):
        self.df = pd.DataFrame(layer_info)
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (15, 8)

    def plot_all(self, save_dir=None):
        """모든 분석 그래프를 그리고 저장"""
        if self.df.empty:
            print("No data to visualize.")
            return

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._plot_weight_stats(save_dir)
        if 'activation_mean' in self.df.columns:
            self._plot_activation_stats(save_dir)
            self._plot_combined(save_dir)

    def _plot_weight_stats(self, save_dir):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
        
        # Mean & Std
        sns.lineplot(data=self.df, x='layer_id', y='weight_mean', label='Mean', ax=ax1, color='b')
        ax1.fill_between(self.df['layer_id'], 
                         self.df['weight_mean'] - self.df['weight_std'],
                         self.df['weight_mean'] + self.df['weight_std'], alpha=0.2, color='b')
        ax1.set_title("Weight Distribution (Mean ± Std)")
        
        # Min & Max
        sns.lineplot(data=self.df, x='layer_id', y='weight_max', label='Max', ax=ax2, color='r', linestyle='--')
        sns.lineplot(data=self.df, x='layer_id', y='weight_min', label='Min', ax=ax2, color='g', linestyle='--')
        ax2.set_title("Weight Range (Min/Max)")
        
        plt.tight_layout()
        if save_dir: plt.savefig(f"{save_dir}/weight_stats.png")
        plt.show()

    def _plot_activation_stats(self, save_dir):
        plt.figure(figsize=(15, 6))
        
        sns.lineplot(data=self.df, x='layer_id', y='activation_mean', label='Mean L2 Norm', color='purple')
        sns.lineplot(data=self.df, x='layer_id', y='activation_max', label='Max L2 Norm', color='orange', linestyle='--')
        
        plt.fill_between(self.df['layer_id'], 
                        self.df['activation_mean'] - self.df['activation_std'],
                        self.df['activation_mean'] + self.df['activation_std'], alpha=0.1, color='purple')
        
        plt.title("Activation Magnitude (L2 Norm) - Check for Outliers")
        plt.ylabel("L2 Norm")
        plt.xlabel("Layer ID")
        
        plt.tight_layout()
        if save_dir: plt.savefig(f"{save_dir}/activation_stats.png")
        plt.show()

    def _plot_combined(self, save_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        sns.boxplot(data=self.df, x='type', y='weight_std', ax=ax1)
        ax1.set_title("Weight Std by Type")
        
        sns.boxplot(data=self.df, x='type', y='activation_mean', ax=ax2)
        ax2.set_title("Activation Mean by Type")
        
        plt.tight_layout()
        if save_dir: plt.savefig(f"{save_dir}/type_comparison.png")
        plt.show()