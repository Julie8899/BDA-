import itertools
import math
import sys

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from plotly import express as px
from plotly import graph_objects as go
from scipy import stats


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):

    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            print("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            print("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def is_cat(df, col_name):
    try:
        pd.to_numeric(df[col_name])
        if len(df[col_name].unique()) < 5:
            return True
        else:
            return False
    except BaseException:
        if (
            len(df[col_name].unique()) == 1
            or len(df[col_name].unique()) / len(df[col_name]) > 0.5
        ):
            raise Exception("Bad Data")
        else:
            return True


def categorize_pred(df, pred_names):
    # make sure to convert all empty string in df to NaN before passing into this function
    cont_pred_names = []
    cat_pred_names = []
    for pred_name in pred_names:
        try:
            if is_cat(df, pred_name):
                cat_pred_names.append(pred_name)
            else:
                cont_pred_names.append(pred_name)
        except BaseException:
            continue

    return cont_pred_names, cat_pred_names


# return df
def get_cont_cont_correlation_table(df, cont_pred_names):
    table_data = []

    for cont1, cont2 in itertools.combinations(cont_pred_names, r=2):
        metric = stats.pearsonr(df[cont1], df[cont2])

        plot_file_name = cont1 + "_" + cont2 + "_correlation.html"
        row_data = [
            cont1,
            cont2,
            metric[0],
            f'<a href="{plot_file_name}">{plot_file_name}</a>',
        ]
        table_data.append(row_data)
        # plot
        fig = px.scatter(x=df[cont1], y=df[cont2], trendline="ols")
        fig.update_layout(
            title=f"{cont1}(Cont) & {cont2}(Cont): PCC = {metric[0]}; p-value = {metric[1]}",
            xaxis_title=cont1,
            yaxis_title=cont2,
        )
        fig.write_html(plot_file_name)

    out_df = pd.DataFrame(
        table_data, columns=["Cont 1", "Cont 2", "PCC", "Plot"]
    ).sort_values("PCC", ascending=False)

    return out_df


# return df
def get_cat_cont_correlation_table(df, cat_pred_names, cont_pred_names, n_bins=20):
    table_data = []

    for cat, cont in itertools.product(cat_pred_names, cont_pred_names):

        metric = cat_cont_correlation_ratio(df[cat], df[cont])
        plot_file_name = cat + "_" + cont + "_correlation.html"
        row_data = [
            cat,
            cont,
            metric,
            f'<a href="{plot_file_name}">{plot_file_name}</a>',
        ]
        table_data.append(row_data)

        unique_cat_values = df[cat].unique().tolist()
        cont_value_groups = []
        calculated_bin_size = math.ceil((df[cont].max() - df[cont].min()) / n_bins)

        for cat_value in unique_cat_values:
            cont_value_group_df = df.loc[df[cat] == cat_value]
            cont_value_groups.append(cont_value_group_df[cont].tolist())

        fig = ff.create_distplot(
            cont_value_groups, unique_cat_values, bin_size=calculated_bin_size
        )
        fig.update_layout(
            title=f"{cat}(Cat) & {cont}(Cont): Correlation Ratio = {metric}",
            xaxis_title=cont,
            legend_title=cat,
            yaxis_title="Distribution",
        )
        fig.write_html(plot_file_name)

    out_df = pd.DataFrame(
        table_data, columns=["Cat", "Cont", "Correlation Ratio", "Plot"]
    ).sort_values("Correlation Ratio", ascending=False)

    return out_df


# return df
def get_cat_cat_correlation_table(df, cat_pred_names):
    data = []

    for cat1, cat2 in itertools.combinations(cat_pred_names, r=2):

        metric = cat_correlation(
            df[cat1], df[cat2], bias_correction=False, tschuprow=False
        )

        plot_file_name = cat1 + "_" + cat2 + "_correlation.html"
        row = [cat1, cat2, metric, f'<a href="{plot_file_name}">{plot_file_name}</a>']
        data.append(row)
        # plot
        unique_cat1_values = df[cat1].unique().tolist()
        unique_cat2_values = df[cat2].unique().tolist()
        grouped = df.groupby([cat1, cat2])

        heatmap_z = []
        for cat2_value in unique_cat2_values:

            heatmap_row = []
            for cat1_value in unique_cat1_values:

                if (cat1_value, cat2_value) not in grouped.groups:
                    heatmap_row.append(0)
                else:
                    heatmap_row.append(
                        len(grouped.groups.get((cat1_value, cat2_value)))
                    )
            heatmap_z.append(heatmap_row)

        fig = ff.create_annotated_heatmap(
            heatmap_z, x=unique_cat1_values, y=unique_cat2_values
        )
        fig.update_layout(
            title=f"{cat1}(Cat) & {cat2}(Cat): Cramer's V = {metric}",
            xaxis_title=cat1,
            yaxis_title=cat2,
        )
        fig.write_html(plot_file_name)

    out_df = pd.DataFrame(
        data, columns=["Cat 1", "Cat 2", "Cramer's V", "Plot"]
    ).sort_values("Cramer's V", ascending=False)

    return out_df


# return html file name
def get_cont_cont_matrix(df, cont_pred_names):
    metric = df[cont_pred_names].corr()
    data = metric.values

    plot_file_name = "cont_cont_matrix.html"
    fig = go.Figure(
        data=go.Heatmap(z=data, x=cont_pred_names, y=cont_pred_names, hoverongaps=False)
    )
    fig.update_layout(title="Cont & Cont Correlation Matrix")
    fig.write_html(plot_file_name)

    return plot_file_name


# return html file name
def get_cat_cont_matrix(df, cat_pred_names, cont_pred_names):
    data = []

    for cont in cont_pred_names:

        row = []

        for cat in cat_pred_names:
            metric = cat_cont_correlation_ratio(df[cat], df[cont])
            row.append(metric)

        data.append(row)

    plot_file_name = "cat_cont_matrix.html"
    fig = go.Figure(
        data=go.Heatmap(z=data, x=cat_pred_names, y=cont_pred_names, hoverongaps=False)
    )
    fig.update_layout(title="Cat & Cont Correlation Matrix")
    fig.write_html(plot_file_name)

    return plot_file_name


# return html file name
def get_cat_cat_matrix(df, cat_pred_names):
    data = []

    for cat1 in cat_pred_names:

        row = []

        for cat2 in cat_pred_names:
            metric = cat_correlation(
                df[cat1], df[cat2], bias_correction=False, tschuprow=False
            )
            row.append(metric)

        data.append(row)

    plot_file_name = "cat_cat_matrix.html"
    fig = go.Figure(
        data=go.Heatmap(z=data, x=cat_pred_names, y=cat_pred_names, hoverongaps=False)
    )
    fig.update_layout(title="Cat & Cat Correlation Matrix")
    fig.write_html(plot_file_name)

    return plot_file_name


# return df
def get_cont_cont_mean_diff_table(df, cont_pred_names, res_name, n_bins=10):
    table_data = []

    n_2d_bins = n_bins * n_bins

    for cont1, cont2 in itertools.combinations(cont_pred_names, r=2):
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

        res_means_diff_weighted_df = res_means_diff_df.mul(res_weights_df)

        diff_mean_unweighted = (res_means_diff_df.pow(2).sum().sum() / n_2d_bins) ** 0.5
        diff_mean_weighted = (
            res_means_diff_weighted_df.pow(2).sum().sum() / n_2d_bins
        ) ** 0.5

        unweighted_plot_file_name = cont1 + "_" + cont2 + "_mean_diff_unweighted.html"
        weighted_plot_file_name = cont1 + "_" + cont2 + "_mean_diff_weighted.html"

        row_data = [
            cont1,
            cont2,
            diff_mean_unweighted,
            f'<a href="{unweighted_plot_file_name}">{unweighted_plot_file_name}</a>',
            diff_mean_weighted,
            f'<a href="{weighted_plot_file_name}">{weighted_plot_file_name}</a>',
        ]
        table_data.append(row_data)

        plot_x = res_means_df.columns.map(lambda x: (x.left + x.right) / 2)
        plot_y = res_means_df.index.map(lambda x: (x.left + x.right) / 2)
        plot_z_unweighted = res_means_diff_df.values
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
        fig_unweighed.write_html(unweighted_plot_file_name)

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
        fig_weighted.write_html(weighted_plot_file_name)

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

    return out_df


# return df
def get_cat_cont_mean_diff_table(
    df, cont_pred_names, cat_pred_names, res_name, n_cont_bins=10
):
    data = []

    for cat, cont in itertools.product(cat_pred_names, cont_pred_names):
        n_2d_bins = n_cont_bins * len(df[cat].unique())

        res_mean = df[res_name].mean()
        res_count = df[res_name].count()

        bins = {}
        bins["cont_bins"] = pd.cut(df[cont], n_cont_bins)
        bins_df = pd.DataFrame(bins)

        bin_joined_df = df.filter([cont, cat, res_name], axis=1).join(bins_df)
        bin_grouped_df = bin_joined_df.groupby(bins_df.columns.to_list() + [cat])
        means_df = bin_grouped_df.mean().unstack(level=0)
        counts_df = bin_grouped_df.count().unstack(level=0)
        res_means_df = means_df[res_name]
        res_means_diff_df = res_means_df.sub(res_mean)
        res_counts_df = counts_df[res_name]
        res_weights_df = res_counts_df.div(res_count)
        res_means_diff_weighted_df = res_means_diff_df.mul(res_weights_df)
        diff_mean_unweighted = (res_means_diff_df.pow(2).sum().sum() / n_2d_bins) ** 0.5
        diff_mean_weighted = (
            res_means_diff_weighted_df.pow(2).sum().sum() / n_2d_bins
        ) ** 0.5

        unweighted_plot_file_name = cat + "_" + cont + "_mean_diff_unweighted.html"
        weighted_plot_file_name = cat + "_" + cont + "_mean_diff_weighted.html"

        row_data = [
            cat,
            cont,
            diff_mean_unweighted,
            f'<a href="{unweighted_plot_file_name}">{unweighted_plot_file_name}</a>',
            diff_mean_weighted,
            f'<a href="{weighted_plot_file_name}">{weighted_plot_file_name}</a>',
        ]
        data.append(row_data)

        plot_x = res_means_df.columns.map(lambda x: (x.left + x.right) / 2)
        plot_y = res_means_df.index
        plot_z_unweighted = res_means_diff_df.values
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
            title=f"{cat}(Cat) & {cont}(Cont): Unweighted Mean Diff = {diff_mean_unweighted}",
            xaxis_title=cont,
            yaxis_title=cat,
        )
        fig_unweighed.write_html(unweighted_plot_file_name)

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
            title=f"{cat}(Cat) & {cont}(Cont): Weighted Mean Diff = {diff_mean_weighted}",
            xaxis_title=cont,
            yaxis_title=cat,
        )
        fig_weighted.write_html(weighted_plot_file_name)

    out_df = pd.DataFrame(
        data,
        columns=[
            "Cat",
            "Cont",
            "Unweighted Mean Diff",
            "Unweighted Plot",
            "Weighted Mean Diff",
            "Wighted Plot",
        ],
    ).sort_values("Weighted Mean Diff", ascending=False)

    return out_df


# return df
def get_cat_cat_mean_diff_table(df, cat_pred_names, res_name):
    table_data = []

    for cat1, cat2 in itertools.combinations(cat_pred_names, r=2):
        n_2d_bins = len(df[cat1].unique()) * len(df[cat2].unique())

        res_mean = df[res_name].mean()
        res_count = df[res_name].count()

        bin_grouped_df = df.filter([cat1, cat2, res_name], axis=1).groupby([cat1, cat2])
        means_df = bin_grouped_df.mean().unstack(level=0)
        counts_df = bin_grouped_df.count().unstack(level=0)
        res_means_df = means_df[res_name]
        res_means_diff_df = res_means_df.sub(res_mean)
        res_counts_df = counts_df[res_name]
        res_weights_df = res_counts_df.div(res_count)
        res_means_diff_weighted_df = res_means_diff_df.mul(res_weights_df)
        diff_mean_unweighted = (res_means_diff_df.pow(2).sum().sum() / n_2d_bins) ** 0.5
        diff_mean_weighted = (
            res_means_diff_weighted_df.pow(2).sum().sum() / n_2d_bins
        ) ** 0.5

        unweighted_plot_file_name = cat1 + "_" + cat2 + "_mean_diff_unweighted.html"
        weighted_plot_file_name = cat1 + "_" + cat2 + "_mean_diff_weighted.html"

        row_data = [
            cat1,
            cat2,
            diff_mean_unweighted,
            f'<a href="{unweighted_plot_file_name}">{unweighted_plot_file_name}</a>',
            diff_mean_weighted,
            f'<a href="{weighted_plot_file_name}">{weighted_plot_file_name}</a>',
        ]
        table_data.append(row_data)

        plot_x = res_means_df.columns
        plot_y = res_means_df.index
        plot_z_unweighted = res_means_diff_df.values
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
            title=f"{cat1}(Cat) & {cat2}(Cat): Unweighted Mean Diff = {diff_mean_unweighted}",
            xaxis_title=cat1,
            yaxis_title=cat2,
        )
        fig_unweighed.write_html(unweighted_plot_file_name)

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
            title=f"{cat1}(Cat) & {cat2}(Cat): Weighted Mean Diff = {diff_mean_weighted}",
            xaxis_title=cat1,
            yaxis_title=cat2,
        )
        fig_weighted.write_html(weighted_plot_file_name)

    out_df = pd.DataFrame(
        table_data,
        columns=[
            "Cat 1",
            "Cat 2",
            "Unweighted Mean Diff",
            "Unweighted Plot",
            "Weighted Mean Diff",
            "Wighted Plot",
        ],
    ).sort_values("Weighted Mean Diff", ascending=False)

    return out_df


def main():
    # INPUT
    pred_names = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    res_name = "Churn"
    # pred_names = [
    #     "customerID",
    #     "gender",
    #     "SeniorCitizen",
    #     "Partner",
    #     "Dependents",
    #     "tenure",
    #     "PhoneService",
    #     "MultipleLines",
    #     "InternetService",
    #     "OnlineSecurity",
    #     "OnlineBackup",
    #     "DeviceProtection",
    #     "TechSupport",
    #     "StreamingTV",
    #     "StreamingMovies",
    #     "Contract",
    #     "PaperlessBilling",
    #     "PaymentMethod",
    #     "MonthlyCharges",
    # ]
    # res_name = 'TotalCharges'
    data_file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Categorize predictors and fix bad data
    print("Categorize predictors and fix bad data")
    df = pd.read_csv(data_file_name)
    df = df.replace(r"^\s*$", np.NaN, regex=True)
    cont_pred_names, cat_pred_names = categorize_pred(df, pred_names)
    for name in cont_pred_names:
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0)

    # correlation tables
    print("Calculating Cont Cont Correlation Table")
    cont_cont_correlation_table_df = get_cont_cont_correlation_table(
        df, cont_pred_names
    )
    cont_cont_correlation_table_html = cont_cont_correlation_table_df.to_html(
        escape=False
    )

    print("Calculating Cat Cont Correlation Table")
    cat_cont_correlation_table_df = get_cat_cont_correlation_table(
        df, cat_pred_names, cont_pred_names
    )
    cat_cont_correlation_table_html = cat_cont_correlation_table_df.to_html(
        escape=False
    )

    print("Calculating Cat Cat Correlation Table")
    cat_cat_correlation_table_df = get_cat_cat_correlation_table(df, cat_pred_names)
    cat_cat_correlation_table_html = cat_cat_correlation_table_df.to_html(escape=False)

    # matrices
    print("Calculating Cont Cont Correlation Matrix")
    cont_cont_matrix_html = get_cont_cont_matrix(df, cont_pred_names)

    print("Calculating Cat Cont Correlation Matrix")
    cat_cont_matrix_html = get_cat_cont_matrix(df, cat_pred_names, cont_pred_names)

    print("Calculating Cat Cat Correlation Matrix")
    cat_cat_matrix_html = get_cat_cat_matrix(df, cat_pred_names)

    # mean diff tables
    mean_diff_tables_html = ""
    if is_cat(df, res_name):
        print(
            "Response is Categorical, no need to calculate Brute-Force Mean Diff Tables"
        )
        mean_diff_tables_html = f"<h2>Cannot calculate Mean Diff because the response ({res_name}) is Categorical.</h2>"
    else:
        df[res_name] = pd.to_numeric(df[res_name], errors="coerce").fillna(0)

        print("Response is Continuous, start calculating Brute-Force Mean Diff Tables")
        mean_diff_tables_html += "<h2>Brute-Force Mean Diff Tables</h2>"

        print("Calculating Cont Cont Brute-Force Mean Diff Table")
        mean_diff_tables_html += "<h3>Cont & Cont Brute-Force Mean Diff Table</h3>"
        cont_cont_mean_diff_table_df = get_cont_cont_mean_diff_table(
            df, cont_pred_names, res_name
        )
        mean_diff_tables_html += cont_cont_mean_diff_table_df.to_html(escape=False)

        print("Calculating Cat Cont Brute-Force Mean Diff Table")
        mean_diff_tables_html += "<h3>Cat & Cont Brute-Force Mean Diff Table</h3>"
        cat_cont_mean_diff_table_df = get_cat_cont_mean_diff_table(
            df, cont_pred_names, cat_pred_names, res_name
        )
        mean_diff_tables_html += cat_cont_mean_diff_table_df.to_html(escape=False)

        print("Calculating Cat Cat Brute-Force Mean Diff Table")
        mean_diff_tables_html += "<h3>Cat & Cat Brute-Force Mean Diff Table</h3>"
        cat_cat_mean_diff_table_df = get_cat_cat_mean_diff_table(
            df, cat_pred_names, res_name
        )
        mean_diff_tables_html += cat_cat_mean_diff_table_df.to_html(escape=False)

    # Write HTML
    print("Writing Result HTML")
    f = open("results.html", "w")
    html = f"""
        <html>
            <head></head>
            <body>
                <h2>Correlation Tables</h2>
                <h3>Cont & Cont Correlation Tables</h3>
                {cont_cont_correlation_table_html}
                <h3>Cat & Cont Correlation Tables</h3>
                {cat_cont_correlation_table_html}
                <h3>Cat & Cat Correlation Tables</h3>
                {cat_cat_correlation_table_html}

                <h2>Correlation Matrices</h2>
                <h3>Cont & Cont Correlation Matrix</h3>
                <a href='{cont_cont_matrix_html}'>Cont & Cont Correlation Matrix</a>
                <h3>Cat & Cont Correlation Matrix</h3>
                <a href='{cat_cont_matrix_html}'>Cat & Cont Correlation Matrix</a>
                <h3>Cat & Cat Correlation Matrix</h3>
                <a href='{cat_cat_matrix_html}'>Cat & Cat Correlation Matrix</a>

                {mean_diff_tables_html}
            </body>
        </html>
    """

    f.write(html)
    f.close()


if __name__ == "__main__":
    sys.exit(main())
