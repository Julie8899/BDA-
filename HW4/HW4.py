import math

import numpy as np
import pandas
import plotly.figure_factory as ff

# import pydot
import statsmodels.api

# from matplotlib import pyplot as plt
# from pandas import DataFrame
from plotly import express as px
from plotly import graph_objects as go

# import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from scipy import stats

# from io import StringIO

# from sklearn import datasets, tree
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV
# from sklearn.tree import DecisionTreeClassifier, export_graphviz

# import tempfile
# from PIL import Image

data_df = pandas.read_csv("suspectheartrates.csv")


def is_column_cat(df, column_name):
    return df[column_name].nunique() <= 2


def plot_heatmap(df, pred_col_name, res_col_name):
    unique_pred_values = df[pred_col_name].unique().tolist()
    unique_res_values = df[res_col_name].unique().tolist()
    grouped = df.groupby([pred_col_name, res_col_name])
    heatmap_z = []
    for res_value in unique_res_values:
        row = []
        for pred_value in unique_pred_values:
            row.append(len(grouped.groups.get((pred_value, res_value))))
        heatmap_z.append(row)
    fig = ff.create_annotated_heatmap(
        heatmap_z, x=unique_pred_values, y=unique_res_values
    )
    fig.update_layout(
        xaxis_title=pred_col_name,
        yaxis_title=res_col_name,
    )
    fig.show()
    return


def plot_grouped_hist(df, cont_col_name, cat_col_name, n_bins=50):
    unique_cat_values = df[cat_col_name].unique().tolist()
    cont_value_groups = []
    calculated_bin_size = math.ceil(
        (data_df[cont_col_name].max() - df[cont_col_name].min()) / n_bins
    )
    print("Bin Size: " + str(calculated_bin_size))
    for cat_value in unique_cat_values:
        cont_value_group_df = df.loc[df[cat_col_name] == cat_value]
        cont_value_groups.append(cont_value_group_df[cont_col_name].tolist())
    fig = ff.create_distplot(
        cont_value_groups, unique_cat_values, bin_size=calculated_bin_size
    )
    fig.update_layout(
        xaxis_title=cont_col_name,
        legend_title=cat_col_name,
        yaxis_title="Distribution",
    )
    fig.show()
    return


def plot_scatter(df, pred_col_name, res_col_name):
    pred_values = df[pred_col_name].tolist()
    res_values = df[res_col_name].tolist()

    predictor = statsmodels.api.add_constant(pred_values)
    linear_regression_model = statsmodels.api.OLS(res_values, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()

    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    fig = px.scatter(x=pred_values, y=res_values, trendline="ols")
    fig.update_layout(
        title=f"Variable: {pred_col_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=pred_col_name,
        yaxis_title=res_col_name,
    )
    fig.show()
    return


def plot_mean_sq_diff(data_df, pred_col_name, res_col_name, weighted=False, n_bins=20):
    pred_values = data_df[pred_col_name].tolist()
    res_values = data_df[res_col_name].tolist()

    binned_counts, bin_edges, bin_nums = stats.binned_statistic(
        pred_values, res_values, "count", bins=n_bins
    )
    binned_means, bin_edges, bin_nums = stats.binned_statistic(
        pred_values, res_values, "mean", bins=n_bins
    )

    res_mean = sum(res_values) / len(res_values)

    binned_mean_sq_diffs = (binned_means - res_mean) ** 2

    title = f"Mean Square Diff of {res_col_name} VS Binned {pred_col_name}: "
    secondary_y_title = f"Square Diff of {res_col_name}"
    if weighted:
        weights = binned_counts / len(pred_values)
        binned_mean_sq_diffs = binned_mean_sq_diffs * weights
        title = "Weighted " + title
        secondary_y_title = "Weighted" + secondary_y_title

    res_mean_sq_diff = np.nansum(binned_mean_sq_diffs) / 10

    title += str(res_mean_sq_diff)

    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=bin_centers, y=binned_counts), secondary_y=False)
    fig.add_trace(go.Scatter(x=bin_centers, y=binned_mean_sq_diffs), secondary_y=True)

    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text=f"{pred_col_name}")
    fig.update_yaxes(title_text=f"{pred_col_name} Distribution", secondary_y=False)
    fig.update_yaxes(title_text=secondary_y_title, secondary_y=True)
    fig.show()

    return


# HW4.py
# Ask for user input
# Input data file name: suspectheartrates.csv
# Input comma separated predictor column names:
# Input response column name:

data_df = pandas.read_csv("suspectheartrates.csv")
# pred_col_names = ["heartbpm", "age", "sex"]
# res_col_name = "veracity"
pred_col_names = ["veracity", "age", "sex"]
res_col_name = "heartbpm"

res_is_cat = is_column_cat(data_df, res_col_name)
for pred_col_name in pred_col_names:
    if res_is_cat:
        if is_column_cat(data_df, pred_col_name):
            # Pred is cat and Res is cat, plot heatmap
            plot_heatmap(data_df, pred_col_name, res_col_name)
        else:
            # Pred is cont and Res is cat, plot grouped hist
            # Calculate p-value and t-score, Logistic Regression
            plot_grouped_hist(
                data_df,
                cont_col_name=pred_col_name,
                cat_col_name=res_col_name,
                n_bins=50,
            )
            pass
    else:
        if is_column_cat(data_df, pred_col_name):
            # Pred is cat and Res is cont, plot grouped hist
            plot_grouped_hist(
                data_df,
                cont_col_name=res_col_name,
                cat_col_name=pred_col_name,
                n_bins=50,
            )
            pass
        else:
            # Pred is cont and Res is cont, plot scatter with trendline
            # Calculate p-value and t-score, Linear Regression
            plot_scatter(data_df, pred_col_name, res_col_name)
            plot_mean_sq_diff(
                data_df, pred_col_name, res_col_name, weighted=False, n_bins=15
            )
            plot_mean_sq_diff(
                data_df, pred_col_name, res_col_name, weighted=True, n_bins=30
            )
            pass
