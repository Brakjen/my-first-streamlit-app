import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplot
import matplotlib.dates as mdates
mplot.style.use("dark_background")

from scipy.optimize import curve_fit


def linear_model(x, a, b):
    return a*x + b


class WeightLoss:
    def __init__(self, data_path, goal=97):
        self.data_path = data_path
        self.goal = goal
        self.data = self.load_data()

        self.projection_length = datetime.date(2023, 12, 31)
        self.projected_range = pd.date_range(start=datetime.date(2023, 1, 1), end=self.projection_length)

    def load_data(self):
        df = pd.read_excel(self.data_path, index_col=0)
        df["Mean"] = (df.Morning + df.Evening) / 2
        df["Week"] = df.index.map(lambda x: x.strftime("%W"))
        return df.dropna()

    def fit_trend(self):
        x = range(self.data.shape[0])
        y = self.data.Mean
        return curve_fit(linear_model, x, y)
    
    def project_trend(self):
        popt, pcov = self.fit_trend()
        xs = self.projected_range
        ys = linear_model(range(xs.shape[0]), *popt)
        return xs, ys
    
    def weight_loss_plan(self):
        factor = 0.3
        scaling = 0.015

        initial = self.data.Mean[0]
        bad_line = np.linspace(initial, self.goal * 1.05, pd.Timestamp(2023, 12, 31).dayofyear)
        good_line = np.linspace(initial, self.goal, pd.Timestamp(2023, 12, 31).dayofyear)
        best_line = np.linspace(initial, self.goal * 0.95, pd.Timestamp(2023, 12, 31).dayofyear)

        bad_upper = bad_line + factor * np.array([i for i in range(len(bad_line))]) * scaling
        bad_lower = bad_line - factor * np.array([i for i in range(len(bad_line))]) * scaling

        good_upper = good_line + factor * np.array([i for i in range(len(good_line))]) * scaling
        good_lower = good_line - factor * np.array([i for i in range(len(good_line))]) * scaling

        best_upper = best_line + factor * np.array([i for i in range(len(best_line))]) * scaling
        best_lower = best_line - factor * np.array([i for i in range(len(best_line))]) * scaling

        return ((bad_lower, bad_upper), (good_lower, good_upper), (best_lower, best_upper))


    def plot(self):
        fig, ax = mplot.subplots(figsize=(8, 4), dpi=100)

        ax.fill_between(self.data.index, self.data.Morning, self.data.Evening, facecolor="skyblue", alpha=0.5, label="Weight")
        ax.plot(*self.project_trend(), color="teal", ls="--", lw=1, label="Projected Trend")

        colors = ["orange", "yellow", "lightgreen"]
        labels = ["Bad", "Good", "Best"]
        for color, label, (lower, upper) in zip(colors, labels, self.weight_loss_plan()):
            ax.fill_between(self.projected_range, lower, upper, facecolor=color, alpha=0.5, label=f"{label} Progress")

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        ax.set_xlim(datetime.date(2023, 1, 1), self.projection_length)
        ax.set_xlabel("2023")
        ax.set_ylabel("Weight (kg)")
        ax.set_title("Weight Plan 2023")
        ax.legend(loc="lower left")

        ax.grid(ls=":", lw=0.5, color="gray")

        fig.tight_layout()
        fig.savefig("plot.png")


if __name__ == "__main__":
    wl = WeightLoss("data.xlsx")
    wl.plot()
