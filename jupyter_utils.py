import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from copy import copy
from matplotlib.colors import ListedColormap


def display_test(df, columns_to_test, target_column, test, nan_policy='omit', stat_name='x', alpha=0.05):
    """
    shows a stats test result for a given list of columns with a key column

    :param df: pd.DataFrame
    :param columns_to_test: list of str, column names
    :param target_column: str, column name to test all others against
    :param test: func, stats test taking lists as inputs and returning a tuple of float, power and significance
    :param nan_policy: str, argument for the test, optional, default 'omit'
    :param stat_name: str, the name of the test variable, optional, default 'x'
    :param alpha: float, significance level, optional, default 0.05

    :return: styled pd.DataFrame with test results
    """
    res_df = pd.DataFrame(columns=[stat_name, 'p', 'sig', f'abs_{stat_name}'], index=columns_to_test)

    for column in columns_to_test:
        if column == target_column:
            r, p = np.nan, np.nan
        elif nan_policy:
            section = df[[target_column, column]].dropna()
            if section.empty:
                r, p = np.nan, np.nan
            else:
                r, p = list(test(section[target_column], section[column]))
        else:
            raise ValueError('correlation does not allow nan propagation')

        res_df.loc[[column]] = [r, p, p < alpha, abs(r)]

    return res_df.sort_values([f'abs_{stat_name}', 'p'], ascending=False).apply(
        pd.to_numeric).style.background_gradient(axis=0, cmap='Reds'), res_df


def display_group_test(df, columns_to_test, target_column, test, nan_policy='omit', stat_name='x', alpha=0.05,
                       group_names=None):
    """
    shows a stats test result for a given list of columns with a key column

    :param df: pd.DataFrame
    :param columns_to_test: list of str, column names
    :param target_column: str, column name to test all others against
    :param test: func, stats test taking lists as inputs and returning a tuple of float, power and significance
    :param nan_policy: str, argument for the test, optional, default 'omit'
    :param stat_name: str, the name of the test variable, optional, default 'x'
    :param alpha: float, significance level, optional, default 0.05
    :param group_names: list of str, optional, default None

    :return: styled pd.DataFrame with test results
    """
    res_df = pd.DataFrame(columns=[stat_name, 'p', 'sig', f'abs_{stat_name}'], index=columns_to_test)

    if group_names is None:
        group_names = df[target_column].dropna().unique().tolist()
        assert len(group_names) == 2, 'only two group tests are supported'

    for column in columns_to_test:
        r, p = list(test(df[df[target_column] == group_names[1]][column],
                         df[df[target_column] == group_names[0]][column], nan_policy=nan_policy))
        res_df.loc[[column]] = [r, p, p < alpha, abs(r)]

    #     reject, pvalscorr = multipletests(res['p'], alpha=alpha, method='b')[:2]
    #     res['bonf'], res['bonf_sig'] = pvalscorr, reject
    #     'bonf', 'bonf_sig'
    #     res_df = pd.DataFrame([list(res.loc[column]) for column in columns_to_test],
    #                            columns=[stat_name, 'p', 'sig'], index=columns_to_test)
    return res_df.sort_values([f'abs_{stat_name}', 'p'], ascending=False).apply(
        pd.to_numeric).style.background_gradient(axis=0, cmap='Reds'), res_df


def color_nan_white(val):
    """Color the nan text white"""
    if np.isnan(val):
        return 'color: white'


def color_nan_white_background(val):
    """Color the nan cell background white"""
    if np.isnan(val):
        return 'background-color: white'


CMAP = copy(plt.cm.get_cmap("Blues"))
CMAP.set_under("white")


def style(df, vmin=None, vmax=None, cmap=CMAP):
    return (df.style
            .background_gradient(vmin=vmin, vmax=vmax, cmap=cmap)
            .applymap(lambda x: color_nan_white(x))
            .applymap(lambda x: color_nan_white_background(x))
            )


def mean_std(df_data, columns, group=None):
    data = df_data.groupby(group)[columns] if group else df_data[columns]
    if group:
        df_mean, df_std = data.mean().round(2), data.std().round(2)

        df = pd.DataFrame(columns=df_mean.columns, index=df_mean.index)
        columns = df_mean.columns.tolist()
        for i, row in df_mean.iterrows():
            for c in columns:
                df.loc[i, c] = f'{df_mean.loc[i, c]} ({df_std.loc[i, c]})'
    else:
        s_mean, s_std = data.mean().round(2), data.std().round(2)
        df = pd.DataFrame(index=s_mean.index, columns=['value'])
        for i, v in s_mean.items():
            df.loc[i, 'value'] = f'{v} ({s_std.loc[i]})'
        df = df.transpose()
    return df


def scatter_annotate(x, y, labels, x_name, y_name, title, c=None,
                     cmap_colors=('indigo', 'steelblue', 'mediumseagreen', 'gold'),
                     classes=('LM', 'graph', 'lexical', 'syntactic')):
    assert len(cmap_colors) == len(classes)
    cmap = ListedColormap(cmap_colors)

    if c is not None:
        scatter = plt.scatter(x, y, c=c, cmap=cmap)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    else:
        plt.scatter(x, y)

    for i, txt in enumerate(labels):
        plt.annotate(txt, (x[i], y[i]))

    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.title(title)


def calculate_pvalues(df_):
    df = df_.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
    return pvalues


def show_corrtest(df, threshold=0.05):
    p_values = calculate_pvalues(df)
    mask = p_values > threshold
    plt.figure(figsize=(40, 20))
    corr = df.corr()
    sns.heatmap(corr, annot=True, mask=mask)
    plt.yticks(rotation=0)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()  # ta-da!;
    return p_values, corr


def show_corrtest_mask_corr(df, threshold=0.5):
    plt.figure(figsize=(40, 20))
    corr = df.corr()
    mask = corr.abs() < threshold
    sns.heatmap(corr, annot=True, mask=mask, cmap='RdBu_r')
    plt.yticks(rotation=0)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()
    return corr
