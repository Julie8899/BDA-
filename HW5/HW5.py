import itertools
import math
import os
import sys

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import sqlalchemy
import statsmodels.api
from plotly import express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def get_p_t_value_ranking_table(df, feature_names, response_name):
    file_path = "result_plots/p_t_ranking"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    p_table_data = []
    t_table_data = []

    for feature_name in feature_names:

        # Calculate p-value
        feature_values = df[feature_name].tolist()
        response_values = df[response_name].tolist()

        predictor = statsmodels.api.add_constant(feature_values)
        linear_regression_model = statsmodels.api.OLS(response_values, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()

        t_value = abs(round(linear_regression_model_fitted.tvalues[1], 6))
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        plot_file_name = feature_name + "_hist"
        plot_file_path = file_path + "/" + plot_file_name + ".html"

        p_row_data = [
            feature_name,
            p_value,
            f'<a href="{plot_file_path}">{plot_file_name}</a>',
        ]
        p_table_data.append(p_row_data)

        t_row_data = [
            feature_name,
            t_value,
            f'<a href="{plot_file_path}">{plot_file_name}</a>',
        ]
        t_table_data.append(t_row_data)

        # Plot
        unique_response_values = df[response_name].unique().tolist()
        feature_value_groups = []
        calculated_bin_size = math.ceil(
            (df[feature_name].max() - df[feature_name].min()) / 20
        )

        for unique_response_value in unique_response_values:
            feature_value_group_df = df.loc[df[response_name] == unique_response_value]
            feature_value_groups.append(feature_value_group_df[feature_name].tolist())

        fig = ff.create_distplot(
            feature_value_groups, unique_response_values, bin_size=calculated_bin_size
        )
        fig.update_layout(
            title=f"t-value = {t_value}; p-value = {p_value}",
            xaxis_title=feature_name,
            legend_title=response_name,
            yaxis_title="Distribution",
        )
        fig.write_html(plot_file_path)

    p_out_df = pd.DataFrame(p_table_data, columns=["Feature", "p-value", "Histogram"])
    t_out_df = pd.DataFrame(
        t_table_data, columns=["Feature", "t-value", "Histogram"]
    ).sort_values("t-value", ascending=True)

    p_out_df["p-value"] = pd.to_numeric(p_out_df["p-value"])
    p_out_df = p_out_df.sort_values("p-value", ascending=False)

    return p_out_df.to_html(escape=False, index=False), t_out_df.to_html(
        escape=False, index=False
    )


def get_diff_mean_ranking_table(df, feature_names, response_name, n_bins=20):
    file_path = "result_plots/mean_diff_ranking"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    table_data = []

    for feature_name in feature_names:
        # Calculate mean-diff
        feature_values = df[feature_name].tolist()
        response_values = df[response_name].tolist()

        binned_counts, bin_edges, bin_nums = stats.binned_statistic(
            feature_values, response_values, "count", bins=n_bins
        )
        binned_means, bin_edges, bin_nums = stats.binned_statistic(
            feature_values, response_values, "mean", bins=n_bins
        )

        response_mean = sum(response_values) / len(response_values)
        unweighted_binned_mean_sq_diffs = (binned_means - response_mean) ** 2
        weights = binned_counts / len(feature_values)
        weighted_binned_mean_sq_diffs = unweighted_binned_mean_sq_diffs * weights

        unweighted_mean_sq_diff = np.nansum(unweighted_binned_mean_sq_diffs) / n_bins
        weighted_mean_sq_diff = np.nansum(weighted_binned_mean_sq_diffs) / n_bins

        unweighted_plot_file_name = feature_name + "_mean_diff_unweighted"
        unweighted_plot_file_path = (
            file_path + "/" + unweighted_plot_file_name + ".html"
        )

        weighted_plot_file_name = feature_name + "_mean_diff_weighted"
        weighted_plot_file_path = file_path + "/" + weighted_plot_file_name + ".html"

        row_data = [
            feature_name,
            unweighted_mean_sq_diff,
            f'<a href="{unweighted_plot_file_path}">{unweighted_plot_file_name}</a>',
            weighted_mean_sq_diff,
            f'<a href="{weighted_plot_file_path}">{weighted_plot_file_name}</a>',
        ]
        table_data.append(row_data)

        # Plot
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2

        # Unweighted
        unweighted_fig = make_subplots(specs=[[{"secondary_y": True}]])
        unweighted_fig.add_trace(
            go.Bar(x=bin_centers, y=binned_counts, name=f"{feature_name}"),
            secondary_y=False,
        )
        unweighted_fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=unweighted_binned_mean_sq_diffs,
                name="Response Binned Mean Difference",
            ),
            secondary_y=True,
        )

        unweighted_fig.update_layout(
            title_text=f"Unweighted Mean Diff of {response_name} VS Binned {feature_name}"
        )

        unweighted_fig.update_yaxes(
            title_text=f"{feature_name} Distribution", secondary_y=False
        )
        unweighted_fig.update_yaxes(
            title_text=f"Unweighted Mean Diff of {response_name}", secondary_y=True
        )

        unweighted_fig.write_html(unweighted_plot_file_path)

        # Weighted
        weighted_fig = make_subplots(specs=[[{"secondary_y": True}]])
        weighted_fig.add_trace(
            go.Bar(x=bin_centers, y=binned_counts, name=f"{feature_name}"),
            secondary_y=False,
        )
        weighted_fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=weighted_binned_mean_sq_diffs,
                name="Response Binned Mean Difference",
            ),
            secondary_y=True,
        )

        weighted_fig.update_layout(
            title_text=f"Weighted Mean Diff of {response_name} VS Binned {feature_name}"
        )

        weighted_fig.update_yaxes(
            title_text=f"{feature_name} Distribution", secondary_y=False
        )
        weighted_fig.update_yaxes(
            title_text=f"Weighted Mean Diff of {response_name}", secondary_y=True
        )

        weighted_fig.write_html(weighted_plot_file_path)

    out_df = pd.DataFrame(
        table_data,
        columns=[
            "Feature",
            "Unweighted Mean Diff",
            "Unweighted Plot",
            "Weighted Mean Diff",
            "Weighted Plot",
        ],
    ).sort_values("Weighted Mean Diff", ascending=False)

    return out_df.to_html(escape=False, index=False)


def get_cont_cont_correlation_table(df, cont_pred_names):
    file_path = "result_plots/correlation"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    table_data = []

    for cont1, cont2 in itertools.combinations(cont_pred_names, r=2):
        print(f"Calculating {cont1} and {cont2} Correlation...")

        metric = stats.pearsonr(df[cont1], df[cont2])

        plot_file_name = cont1 + "_" + cont2 + "_correlation"
        plot_file_path = file_path + "/" + plot_file_name + ".html"
        row_data = [
            cont1,
            cont2,
            metric[0],
            f'<a href="{plot_file_path}">{plot_file_name}</a>',
        ]
        table_data.append(row_data)
        # plot
        fig = px.scatter(x=df[cont1], y=df[cont2], trendline="ols")
        fig.update_layout(
            title=f"{cont1}(Cont) & {cont2}(Cont): PCC = {metric[0]}",
            xaxis_title=cont1,
            yaxis_title=cont2,
        )
        fig.write_html(plot_file_path)

    out_df = pd.DataFrame(
        table_data, columns=["Cont 1", "Cont 2", "PCC", "Plot"]
    ).sort_values("PCC", ascending=False)

    return out_df.to_html(escape=False, index=False)


def get_cont_cont_matrix(df, cont_pred_names):

    file_path = "result_plots"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    metric = df[cont_pred_names].corr()
    data = metric.values

    plot_file_name = "cont_cont_matrix"
    plot_file_path = file_path + "/" + plot_file_name
    html_path = plot_file_path + ".html"
    image_path = plot_file_path + ".png"
    fig = go.Figure(
        data=go.Heatmap(z=data, x=cont_pred_names, y=cont_pred_names, hoverongaps=False)
    )
    fig.update_layout(title="Correlation Matrix")
    fig.write_html(html_path)
    fig.write_image(image_path)

    return html_path, image_path


def get_cont_cont_mean_diff_table(df, cont_pred_names, res_name, n_bins=10):

    file_path = "result_plots/brute-force"
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    table_data = []

    n_2d_bins = n_bins * n_bins

    for cont1, cont2 in itertools.combinations(cont_pred_names, r=2):
        print(f"Calculating {cont1} and {cont2} Binned Mean Diff...")

        res_mean = df[res_name].mean()
        res_count = df[res_name].count()

        bins = {}
        bins["cont1_bins"] = pd.cut(df[cont1], n_bins)
        bins["cont2_bins"] = pd.cut(df[cont2], n_bins)
        bins_df = pd.DataFrame(bins)
        bin_joined_df = df.filter([cont1, cont2, res_name], axis=1).join(bins_df)
        bin_grouped_df = bin_joined_df.groupby(bins_df.columns.to_list())
        means_df = bin_grouped_df.mean().unstack(level=0)
        counts_df = bin_grouped_df.count().unstack(level=0)
        res_means_df = means_df[res_name]
        res_means_diff_df = res_means_df.sub(res_mean)
        res_counts_df = counts_df[res_name]
        res_weights_df = res_counts_df.div(res_count)

        res_means_diff_unweighted_df = res_means_diff_df.pow(2)
        res_means_diff_weighted_df = res_means_diff_unweighted_df.mul(res_weights_df)

        diff_mean_unweighted = res_means_diff_unweighted_df.sum().sum() / n_2d_bins
        diff_mean_weighted = res_means_diff_weighted_df.sum().sum() / n_2d_bins

        unweighted_plot_file_name = cont1 + "_" + cont2 + "_mean_diff_unweighted"
        unweighted_plot_file_path = (
            file_path + "/" + unweighted_plot_file_name + ".html"
        )
        weighted_plot_file_name = cont1 + "_" + cont2 + "_mean_diff_weighted"
        weighted_plot_file_path = file_path + "/" + weighted_plot_file_name + ".html"

        row_data = [
            cont1,
            cont2,
            diff_mean_unweighted,
            f'<a href="{unweighted_plot_file_path}">{unweighted_plot_file_name}</a>',
            diff_mean_weighted,
            f'<a href="{weighted_plot_file_path}">{weighted_plot_file_name}</a>',
        ]
        table_data.append(row_data)

        plot_x = res_means_df.columns.map(lambda x: (x.left + x.right) / 2)
        plot_y = res_means_df.index.map(lambda x: (x.left + x.right) / 2)
        plot_z_unweighted = res_means_diff_unweighted_df.values
        plot_z_weighted = res_means_diff_weighted_df.values

        fig_unweighed = go.Figure(
            data=go.Heatmap(
                z=plot_z_unweighted,
                x=plot_x,
                y=plot_y,
                colorbar={"title": res_name},
                hoverongaps=False,
            )
        )
        fig_unweighed.update_layout(
            title=f"{cont1}(Cont) & {cont2}(Cont): Unweighted Mean Diff = {diff_mean_unweighted}",
            xaxis_title=cont1,
            yaxis_title=cont2,
        )
        fig_unweighed.write_html(unweighted_plot_file_path)

        fig_weighted = go.Figure(
            data=go.Heatmap(
                z=plot_z_weighted,
                x=plot_x,
                y=plot_y,
                colorbar={"title": res_name},
                hoverongaps=False,
            )
        )
        fig_weighted.update_layout(
            title=f"{cont1}(Cont) & {cont2}(Cont): Weighted Mean Diff = {diff_mean_weighted}",
            xaxis_title=cont1,
            yaxis_title=cont2,
        )
        fig_weighted.write_html(weighted_plot_file_path)

    out_df = pd.DataFrame(
        table_data,
        columns=[
            "Cont 1",
            "Cont 2",
            "Unweighted Mean Diff",
            "Unweighted Plot",
            "Weighted Mean Diff",
            "Wighted Plot",
        ],
    ).sort_values("Weighted Mean Diff", ascending=False)

    return out_df.to_html(escape=False, index=False)


def get_html_list(list_data):
    html = "<ul>"
    for item in list_data:
        html += "<li>"
        html += item
        html += "</li>"
    html += "</ul>"

    return html


def get_model_scoring_table(
    df, selected_feature_names, response_name, model_types, scoring_types, n_cv=5
):
    X = df[selected_feature_names].values
    y = df[response_name].values

    table_data = []

    for model_type in model_types:

        row_data = []

        model = None
        if model_type == "logistic_regression":
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
        elif model_type == "random_forest":
            model = RandomForestClassifier()
        elif model_type == "best_tree":
            parameters = {
                "max_depth": range(1, 5),
                "criterion": ["gini", "entropy"],
            }
            decision_tree_grid_search = GridSearchCV(
                DecisionTreeClassifier(), parameters, n_jobs=4
            )
            decision_tree_grid_search.fit(X=X, y=y)
            model = decision_tree_grid_search.best_estimator_
        else:
            continue

        for scoring_type in scoring_types:
            print(f"Calculating {model_type} model's {scoring_type} score...")

            scores = cross_val_score(model, X, y, cv=n_cv, scoring=scoring_type)
            avg_score = np.average(scores)
            row_data.append(avg_score)

        table_data.append(row_data)

    out_df = pd.DataFrame(table_data, columns=scoring_types, index=model_types).T

    return out_df.to_html(escape=False)


def main():
    feature_names = [
        "home_final_score_avg",
        "home_off_batting_avg",
        "home_def_batting_avg",
        "home_off_plateapperance_strikeout",
        "home_def_plateapperance_strikeout",
        "home_off_single_avg",
        "home_def_single_avg",
        "home_off_double_avg",
        "home_def_double_avg",
        "home_off_triple_avg",
        "home_def_triple_avg",
        "home_off_walk_to_strikeout_avg",
        "home_def_walk_to_strikeout_avg",
        "home_off_ground_to_flyout_avg",
        "home_def_ground_to_flyout_avg",
        "away_final_score_avg",
        "away_off_batting_avg",
        "away_def_batting_avg",
        "away_off_plateapperance_strikeout",
        "away_def_plateapperance_strikeout",
        "away_off_single_avg",
        "away_def_single_avg",
        "away_off_double_avg",
        "away_def_double_avg",
        "away_off_triple_avg",
        "away_def_triple_avg",
        "away_off_walk_to_strikeout_avg",
        "away_def_walk_to_strikeout_avg",
        "away_off_ground_to_flyout_avg",
        "away_def_ground_to_flyout_avg",
    ]
    response_name = "home_team_win"

    db_user = "root"
    db_pass = "bda696"  # pragma: allowlist secret
    db_host = "localhost:3306"
    db_database = "baseball"
    connect_string = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma: allowlist secret

    query = "SELECT * FROM temp_for_training"
    sql_engine = sqlalchemy.create_engine(connect_string)

    df = pd.read_sql_query(query, sql_engine)
    df = df[feature_names + [response_name]].fillna(0).astype(np.float32)

    # Calculate p_value_ranking_table_html and t_value_ranking_table_html
    (
        p_value_ranking_table_html,
        t_value_ranking_table_html,
    ) = get_p_t_value_ranking_table(df, feature_names, response_name)
    # Calculate diff_mean_table_html
    diff_mean_ranking_table_html = get_diff_mean_ranking_table(
        df, feature_names, response_name
    )

    # Calculate feature_correlation_table_html
    feature_correlation_table_html = get_cont_cont_correlation_table(df, feature_names)

    # Calculate feature_correlation_matrix_file
    (
        feature_correlation_matrix_html,
        feature_correlation_matrix_image,
    ) = get_cont_cont_matrix(df, feature_names)

    # Calculate feature_2d_diff_mean_table_html
    feature_2d_diff_mean_table_html = get_cont_cont_mean_diff_table(
        df, feature_names, response_name
    )

    selected_feature_names = [
        "home_final_score_avg",
        "home_off_batting_avg",
        "home_def_batting_avg",
        "home_off_plateapperance_strikeout",
        "home_def_plateapperance_strikeout",
        "home_off_single_avg",
        "home_def_single_avg",
        "home_off_double_avg",
        "home_def_double_avg",
        "home_off_triple_avg",
        "home_def_triple_avg",
        "home_off_walk_to_strikeout_avg",
        "home_def_walk_to_strikeout_avg",
        "home_off_ground_to_flyout_avg",
        "home_def_ground_to_flyout_avg",
        "away_final_score_avg",
        "away_off_batting_avg",
        "away_def_batting_avg",
        "away_off_plateapperance_strikeout",
        "away_def_plateapperance_strikeout",
        "away_off_single_avg",
        "away_def_single_avg",
        "away_off_double_avg",
        "away_def_double_avg",
        "away_off_triple_avg",
        "away_def_triple_avg",
        "away_off_walk_to_strikeout_avg",
        "away_def_walk_to_strikeout_avg",
        "away_off_ground_to_flyout_avg",
        "away_def_ground_to_flyout_avg",
    ]

    model_types = ["logistic_regression", "random_forest", "best_tree"]
    scoring_types = ["accuracy", "roc_auc", "average_precision", "precision"]

    selected_features_html = get_html_list(selected_feature_names)

    model_scoring_table_html = get_model_scoring_table(
        df, selected_feature_names, response_name, model_types, scoring_types
    )

    f = open("results.html", "w")
    html = f"""
        <html>
            <head></head>
            <body>
                <h2>P-Value Ranking (Sorted by p-value DESC)</h2>
                {p_value_ranking_table_html}

                <h2>T-Value Ranking (Sorted by t-value DESC)</h2>
                {t_value_ranking_table_html}
                <p>The chosen features can be used to predict the response which means
                that the distribution will have less overlap for win and lose.</p>
                  <p>According to the plot, home_def_ground_to_flyout_avg, away_def_ground_to_flyout_avg,
                  home_def_walk_to_strikeout_avg,</p>
                  <p>home_off_ground_to_flyout_avg, home_off_plateapperance_strikeout,
                  away_off_plateapperance_strikeout, home_off_walk_to_strikeout_avg,</p>
                  <p>home_def_plateapperance_strikeout, home_off_batting_avg,
                  away_off_batting_avg, away_def_batting_avg, home_def_batting_avg.</p>

                <h2>Mean Diff Ranking (Sorted by Weighted DESC)</h2>
                {diff_mean_ranking_table_html}
                <p>The larger the different with mean,
                predictors have the more influence on response</p>
                <p>Based on this, we can choose the features with larger weighted
                different with mean as predictors which is</p>
                <p>away_final_score_avg, home_def_batting_avg, away_def_batting_avg,
                away_def_batting_avg, home_final_score_avg, home_def_double_avg</p>
                <p>home_def_single_avg, away_off_double_avg, home_off_batting_avg,
                away_off_walk_to_strikeout_avg, away_off_single_avg, away_def_double_avg</p>

                <h2>Correlations</h2>
                <h3>Correlation Table (Sorted by PCC DESC)</h3>
                {feature_correlation_table_html}
                <p>PCC is Pearson correlation coefficient. The higher PCC,
                the more significant relationship between predictors.</p>
                <p>PCC is closer to 0 which means that there are less reationship between predictors.</p>
                <p>Based on this, home_def_batting_avg,
                away_def_ground_to_flyout_avg, home_def_ground_to_flyout_avg,</p>
                <p>home_off_batting_avg, away_off_ground_to_flyout_avg,
                away_def_plateapperance_strikeout, away_off_ground_to_flyout_avg</p>
                <p>away_def_batting_avg, home_off_walk_to_strikeout_avg,
                home_off_ground_to_flyout_avg</p>

                <a href='{feature_correlation_matrix_html}'><img src='{feature_correlation_matrix_image}'/></a>
                <p>The coefficient is same with the correlation table</p>

                <h2>Brute-Force Mean Diff (Sorted by Weighted DESC)</h2>
                {feature_2d_diff_mean_table_html}
                <p>We need to use one table to explore the different with mean
                between one predictor with response and to select the features which influence
                response more than others</p>

                <h2>Feature analysis</h2>
                <p>According to the results I got from each table, I decide to use the following
                feature to fit the model and see the different score</p>
                <p>home_def_ground_to_flyout_avg, away_def_ground_to_flyout_avg,
                home_off_ground_to_flyout_avg, away_off_ground_to_flyout_avg, home_def_walk_to_strikeout_avg,</p>
                <p>home_off_ground_to_flyout_avg, home_off_plateapperance_strikeout,
                away_off_plateapperance_strikeout, home_off_walk_to_strikeout_avg,</p>
                <p>home_def_plateapperance_strikeout, home_off_batting_avg, away_off_batting_avg, away_def_batting_avg,
                home_def_batting_avg</p>
                <p>After running the selected features, the average score is lower than the
                score for all features so I decide to use the all features I have.</p>

                <h2>Selected Features:</h2>
                {selected_features_html}

                <h2>Model Scoring Comparison</h2>
                {model_scoring_table_html}

                <h2>Conclusion</h2>
                <p>Based on Model Scoring Comparison Table, Logistic regression has better
                performace score than other models</p>

            </body>
        </html>
    """

    f.write(html)
    f.close()


if __name__ == "__main__":
    sys.exit(main())
