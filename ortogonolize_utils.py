import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def draw_scatter(X, y, target_col, control_col, test_col, column_names=None, target_name=None):
    if column_names:
        control_col, test_col = column_names
    if target_name:
        target_col = target_name

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    axs[0].scatter(X[control_col], y)
    axs[0].set_xlabel(control_col)
    axs[0].set_ylabel(target_col)
    axs[0].set_title(f'{target_col} vs {control_col}')

    axs[1].scatter(X[test_col], y)
    axs[1].set_xlabel(test_col)
    axs[1].set_ylabel(target_col)
    axs[1].set_title(f'{target_col} vs {test_col}')

    axs[2].scatter(X[control_col], X[test_col])
    axs[2].set_xlabel(control_col)
    axs[2].set_ylabel(test_col)
    axs[2].set_title(f'{control_col} vs {test_col}')

    plt.tight_layout()
    plt.show()


def prepare_data(df, target_col, control_col, test_col, column_names=None, target_name=None, add_sq=False):
    X = df[[control_col, test_col]]
    if column_names and len(column_names) == len(X.columns):
        X.columns = column_names
    y = df[target_col]
    if target_name:
        y.columns = target_name

    na_filt = X.isna().any(axis=1) | y.isna()
    X = X[~na_filt]
    y = y[~na_filt]
    if add_sq:
        sq_name = X.columns[0] + ('_sq',) if isinstance(X.columns[0], tuple) else X.columns[0] + 'sq'
        X[sq_name] = X.iloc[:, 0] ** 2
        X = X[[X.columns[0], X.columns[2], X.columns[1]]]
    return X, y


def compute_coefficient(df, target_col, test_col, column_names=None, target_name=None, add_sq=False):
    X = df[[test_col]]
    if column_names and len(column_names) == len(X.columns):
        X.columns = column_names
    y = df[target_col]
    if target_name:
        y.columns = target_name

    na_filt = X.isna().any(axis=1) | y.isna()
    X = X[~na_filt]
    y = y[~na_filt]
    if add_sq:
        sq_name = X.columns[0] + ('_sq',) if isinstance(X.columns[0], tuple) else X.columns[0] + 'sq'
        X[sq_name] = X.iloc[:, 0] ** 2
        X = X[[X.columns[0], X.columns[2], X.columns[1]]]

    # add intercept
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return model.rsquared, X


def compute_ortogonolized_coefficient(df, target_col, control_col, test_col, column_names=None, target_name=None,
                                      add_sq=False):
    X, y = prepare_data(df, target_col, control_col, test_col, column_names=column_names, target_name=target_name,
                        add_sq=add_sq)

    # add intercept
    X = sm.add_constant(X)

    # perform QR decomposition and rescale
    Q, R = np.linalg.qr(X)
    Z = Q * np.diag(R) + X.mean(axis=0).to_numpy()[None, :]
    Z_fit = Z[:, [0, 3]] if add_sq else Z[:, [0, 2]]
    Z_fit = pd.DataFrame(Z_fit, columns=[X.columns[0], X.columns[-1]], index=X.index)

    model = sm.OLS(y, Z_fit).fit()

    return model.rsquared, pd.DataFrame(Z, columns=X.columns, index=X.index)


def compute_ortogonolized_logit(df, target_col, control_col, test_col, column_names=None, target_name=None):
    X, y = prepare_data(df, target_col, control_col, test_col, column_names=column_names, target_name=target_name)

    # add intercept
    X = sm.add_constant(X)

    # perform QR decomposition and rescale
    Q, R = np.linalg.qr(X)
    Z = Q * np.diag(R)  # + X.mean(axis=0).to_numpy()[None, :]
    Z = pd.DataFrame(Z, columns=X.columns, index=X.index)

    model = sm.Logit(y, Z).fit()

    return model.prsquared


def draw_corrected_scatter(df, target_col, control_col, test_col, column_names=None, target_name=None, add_sq=False):
    X, y = prepare_data(df, target_col, control_col, test_col, column_names=column_names, target_name=target_name,
                        add_sq=add_sq)
    r, Z = compute_ortogonolized_coefficient(df, target_col, control_col, test_col, column_names=column_names,
                                             target_name=target_name, add_sq=add_sq)
    draw_scatter(Z, y, target_col, control_col, test_col, column_names=column_names, target_name=target_name)
