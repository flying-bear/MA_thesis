import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

from copy import copy
from matplotlib.colors import ListedColormap
from random import choices
from scipy import stats


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
    return (df.apply(pd.to_numeric).style
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


def add_grey(axes, r=0.3, line_dir='v', min_=100, max_=-100):
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    min_left = min_
    max_right = max_

    for ax in axes.ravel():
        if line_dir == 'v':
            left, right = ax.get_xlim()
        elif line_dir == 'h':
            left, right = ax.get_ylim()
        else:
            raise ValueError('line_dir must be "v" or "h"')
        left = 0 if left > 0 else left
        right = 0 if right < 0 else right
        min_left = left if min_left > left else min_left
        max_right = right if max_right < right else max_right

    for ax in axes.ravel():
        left = -r if min_left < -r else min_left
        right = r if max_right > r else max_right
        if line_dir == 'v':
            ax.axvspan(left, right, color='grey', alpha=0.15)
            ax.axvline(x=0, color='darkgrey', linestyle='--')
        elif line_dir == 'h':
            ax.axhspan(left, right, color='grey', alpha=0.15)
            ax.axhline(y=0, color='darkgrey', linestyle='--')


def pointplot(data, x, y, hue, ax=None, use_errorbar=False, estimator='median', errorbar=('pi', 50), **kwargs):
    sns.set_theme(style="whitegrid")
    if not use_errorbar:
        sns.pointplot(
            data=data, x=x, y=y, hue=hue,
            ax=ax, **kwargs)
    else:
        sns.pointplot(
            data=data, x=x, y=y, hue=hue,
            ax=ax, estimator=estimator, errorbar=errorbar, dodge=True,
            err_kws={'alpha': 0.3}, capsize=.0, alpha=0.7, **kwargs)


def reorder_synt_idx(df):
    ms = sorted(list(df['metric'].unique()))
    for s in ('syntactic: mean_sent_len', 'mean_sent_len'):
        if s in ms:
            ms.remove(s)
            ms.append(s)
    order = ms
    hue_order = ms
    return order, hue_order


def pointplot_horizontal(df, x, ax, reorder_synt=True, estimator='median', errorbar=('pi', 50), **kwargs):
    if reorder_synt:
        order, hue_order = reorder_synt_idx(df)
    else:
        order, hue_order = None, None
    sns.pointplot(
        data=df, x=x, y="metric",
        errorbar=errorbar, estimator=estimator, capsize=.0,
        linestyle="none", hue="metric", palette='tab20' if len(df['metric'].unique()) >= 10 else 'tab10',
        err_kws={'alpha': 0.3}, alpha=0.7, ax=ax, order=order, hue_order=hue_order, **kwargs)


def pointplot_horizontal_wo_errorbar(df, x, ax, reorder_synt=True, **kwargs):
    if reorder_synt:
        order, hue_order = reorder_synt_idx(df)
    else:
        order, hue_order = None, None
    sns.pointplot(data=df, x=x, y="metric",
                  linestyle="none", hue="metric",
                  palette='tab20' if len(df['metric'].unique()) >= 10 else 'tab10',
                  ax=ax, order=order, hue_order=hue_order, **kwargs)


def prep_horizontal_pointplot_errobar_data(df, col, plot_abs=False):
    r = np.concatenate(df[col].to_numpy())
    if plot_abs:
        r = map(abs, r)
    idx = np.concatenate([[x] * len(df[col][x]) for x in df[col].index])
    if len(idx.shape) > 1:
        idx = list(map(lambda x: x[0] + ': ' + x[1], idx))
    d = pd.DataFrame(data=r, columns=[col])
    d['metric'] = idx
    return d


def map_model(m: str) -> str:
    if 'bert' in m or 'sprob' in m or 'sporb' in m:
        return 'bert'
    elif 'raw' in m:
        return 'w2v_raw'
    else:
        return 'w2v'


def prep_LM_pointplot_data(df, plot_abs=False, map_model_f=None):
    if map_model_f is None:
        map_model_f = lambda x: x.rsplit('_', 1)[0]
    if plot_abs:
        df = df.applymap(abs)
    df['model'] = [map_model_f(m) for m in df.index]
    df['metric'] = [x.split('_')[-1] for x in df.index]
    return df


def prep_LM_pointplot_errobar_data(df, col, plot_abs=False, map_model_f=None):
    if map_model_f is None:
        map_model_f = lambda x: x.rsplit('_', 1)[0]
    r = np.concatenate(df[col].to_numpy())
    if plot_abs:
        r = map(abs, r)
    idx = np.concatenate([[x] * len(df[col][x]) for x in df[col].index])
    d = pd.DataFrame(data=r, columns=[col])
    d['model'] = list(map(map_model_f, idx))
    d['metric'] = list(map(lambda x: x.split('_')[-1], idx))
    return d


def prep_LM_pointplot(df, col, plot_abs=False, use_errorbar=True):
    if not use_errorbar:
        return prep_LM_pointplot_data(df, plot_abs=plot_abs)
    else:
        return prep_LM_pointplot_errobar_data(df, col, plot_abs=plot_abs)


def draw_sample_with_replacement(df, seed=None):
    if seed:
        random.seed(seed)
    length = len(df.index)
    idxs = choices(range(length), k=length)
    return df.iloc[idxs, :]


def t_test(df, column, target_column, test=stats.ttest_ind, nan_policy='omit', stat_name='x', alpha=0.05,
           group_names=None):
    if group_names is None:
        group_names = df[target_column].dropna().unique().tolist()
        assert len(group_names) == 2, 'only two group tests are supported'

    r, p = test(df[df[target_column] == group_names[1]][column],
                df[df[target_column] == group_names[0]][column], nan_policy=nan_policy)
    return r
