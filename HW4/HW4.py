import math
import sys
from io import StringIO

import numpy as np
import pandas
import plotly.figure_factory as ff
import pydot
import statsmodels.api
from pandas import DataFrame
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def is_column_cat(df, column_name):
    return df[column_name].nunique() <= 2


def plot_heatmap(df, pred_name, res_name):
    unique_pred_values = df[pred_name].unique().tolist()
    unique_res_values = df[res_name].unique().tolist()
    grouped = df.groupby([pred_name, res_name])
    heatmap_z = []
    for res_value in unique_res_values:
        row = []
        for pred_value in unique_pred_values:
            row.append(len(grouped.groups.get((pred_value, res_value))))
        heatmap_z.append(row)
    fig = ff.create_annotated_heatmap(
        heatmap_z, x=unique_pred_values, y=unique_res_values
    )
    fig.update_layout(xaxis_title=pred_name, yaxis_title=res_name)
    fig.show()
    return


def plot_grouped_hist(df, cont_col_name, cat_col_name, n_bins=50):
    unique_cat_values = df[cat_col_name].unique().tolist()
    cont_value_groups = []
    calculated_bin_size = math.ceil(
        (df[cont_col_name].max() - df[cont_col_name].min()) / n_bins
    )
    print("Bin Size: " + str(calculated_bin_size))
    for cat_value in unique_cat_values:
        cont_value_group_df = df.loc[df[cat_col_name] == cat_value]
        cont_value_groups.append(cont_value_group_df[cont_col_name].tolist())
    fig = ff.create_distplot(
        cont_value_groups, unique_cat_values, bin_size=calculated_bin_size
    )
    fig.update_layout(
        xaxis_title=cont_col_name, legend_title=cat_col_name, yaxis_title="Distribution"
    )
    fig.show()
    return


def plot_scatter(df, pred_name, res_name):
    pred_values = df[pred_name].tolist()
    res_values = df[res_name].tolist()

    predictor = statsmodels.api.add_constant(pred_values)
    linear_regression_model = statsmodels.api.OLS(res_values, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()

    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    fig = px.scatter(x=pred_values, y=res_values, trendline="ols")
    fig.update_layout(
        title=f"Variable: {pred_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=pred_name,
        yaxis_title=res_name,
    )
    fig.show()
    return


def plot_mean_sq_diff(df, pred_name, res_name, weighted=False, n_bins=20):
    pred_values = df[pred_name].tolist()
    res_values = df[res_name].tolist()

    binned_counts, bin_edges, bin_nums = stats.binned_statistic(
        pred_values, res_values, "count", bins=n_bins
    )
    binned_means, bin_edges, bin_nums = stats.binned_statistic(
        pred_values, res_values, "mean", bins=n_bins
    )

    res_mean = sum(res_values) / len(res_values)

    binned_mean_sq_diffs = (binned_means - res_mean) ** 2

    title = f"Mean Square Diff of {res_name} VS Binned {pred_name}: "
    secondary_y_title = f"Square Diff of {res_name}"
    if weighted:
        weights = binned_counts / len(pred_values)
        binned_mean_sq_diffs = binned_mean_sq_diffs * weights
        title = "Weighted " + title
        secondary_y_title = "Weighted" + secondary_y_title

    res_mean_sq_diff = np.nansum(binned_mean_sq_diffs) / n_bins

    title += str(res_mean_sq_diff)

    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=bin_centers, y=binned_counts), secondary_y=False)
    fig.add_trace(go.Scatter(x=bin_centers, y=binned_mean_sq_diffs), secondary_y=True)

    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text=f"{pred_name}")
    fig.update_yaxes(title_text=f"{pred_name} Distribution", secondary_y=False)
    fig.update_yaxes(title_text=secondary_y_title, secondary_y=True)
    fig.show()

    return


# HW4.py
# Ask for user input
# Input data file name: suspectheartrates.csv
# Input comma separated predictor column names:
# Input response column name:
def main():
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

        else:
            if is_column_cat(data_df, pred_col_name):
                # Pred is cat and Res is cont, plot grouped hist
                plot_grouped_hist(
                    data_df,
                    cont_col_name=res_col_name,
                    cat_col_name=pred_col_name,
                    n_bins=50,
                )

            else:
                # Pred is cont and Res is cont, plot scatter with trendline
                # Calculate p-value and t-score, Linear Regression
                plot_scatter(data_df, pred_col_name, res_col_name)
                plot_mean_sq_diff(
                    data_df, pred_col_name, res_col_name, weighted=False, n_bins=15
                )
                plot_mean_sq_diff(
                    data_df, pred_col_name, res_col_name, weighted=True, n_bins=15
                )

            def plot_decision_tree(decision_tree, feature_names, class_names, file_out):
                with StringIO() as dot_data:
                    export_graphviz(
                        decision_tree,
                        feature_names=feature_names,
                        class_names=class_names,
                        out_file=dot_data,
                        filled=True,
                    )
                    graph = pydot.graph_from_dot_data(dot_data.getvalue())
                    graph[0].write_pdf(
                        file_out + ".pdf"
                    )  # must access graph's first element
                    graph[0].write_png(
                        file_out + ".png"
                    )  # must access graph's first element

            # Continuous Features
            continuous_features = ["heartbpm", "age"]
            X = data_df[continuous_features].values

            # Response
            y = data_df["veracity"].values

            # Decision Tree Classifier
            max_tree_depth = 7
            tree_random_state = 0  # Always set a seed
            decision_tree = DecisionTreeClassifier(
                max_depth=max_tree_depth, random_state=tree_random_state
            )
            decision_tree.fit(X, y)

            plot_decision_tree(
                decision_tree=decision_tree,
                feature_names=continuous_features,
                class_names=["Lie", "Truth"],
                file_out="decisiontree",
            )

    # Find an optimal tree via cross-validation
    parameters = {
        "max_depth": range(1, 5),
        "criterion": ["gini", "entropy"],
    }
    decision_tree_grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=tree_random_state), parameters, n_jobs=4
    )
    decision_tree_grid_search.fit(X=X, y=y)

    cv_results = DataFrame(decision_tree_grid_search.cv_results_["params"])
    cv_results["score"] = decision_tree_grid_search.cv_results_["mean_test_score"]

    print(cv_results)

    return


if __name__ == "__main__":
    sys.exit(main())
