import pandas as pd
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def remove_chars(s, chars):
    cs = set(chars)
    return ''.join([c for c in s if c not in cs])


def str_to_int(x):
    if type(x) is str:
        x = remove_chars(x, ',')

    return int(x)


def is_same(l1, l2):
    if len(l1) != len(l2):
        return False

    for x1, x2 in zip(l1, l2):
        if x1 != x2:
            return False

    return True


def parse_leg(year, key0):
    try:
        fn = f'./{year}/立委/不分區立委-A05-6-得票數一覽表({key0}).xlsx'
        df = pd.read_excel(fn)
    except Exception:
        try:
            fn = f'./{year}/立委/不分區立委-A05-6-得票數一覽表({key0}).xls'
            df = pd.read_excel(fn)
        except Exception:
            fn = f'./{year}/立委/立委-A05-6-({key0}).xls'
            df = pd.read_excel(fn)

    candidate_row = df.iloc[1]
    candidate_indexs = [idx for idx in range(len(candidate_row)) if type(candidate_row.iloc[idx]) is str]
    candidates = [remove_chars(candidate_row.iloc[idx], '\n') for idx in candidate_indexs]

    candidate_indexs += [-7, -5]
    candidates += ['立委_無效票', '立委_已領票未投票']

    ret_df = pd.DataFrame(columns=candidates)

    key1 = None
    key2 = None
    key3 = None
    for idx in range(5, len(df)):
        row = df.iloc[idx]
        x = row.iloc[0]
        y = row.iloc[1]
        z = row.iloc[2]

        votes = [str_to_int(row.iloc[cidx]) for cidx in candidate_indexs]

        if type(x) is str and len(remove_chars(x, '\u3000')) != 0:
            key1 = remove_chars(x, '\u3000')
            continue

        key2 = y
        key3 = z

        key = f'{key0}_{key1}_{key2}_{key3}'
        ret_df.loc[key] = votes

    return ret_df


def parse_president(year, key0):
    try:
        fn = f'./{year}/總統/總統-A05-4-候選人得票數一覽表-各投開票所({key0}).xlsx'
        df = pd.read_excel(fn)
    except Exception:
        try:
            fn = f'./{year}/總統/總統-A05-4-候選人得票數一覽表-各投開票所({key0}).xls'
            df = pd.read_excel(fn)
        except Exception:
            fn = f'./{year}/總統/總統-A05-4-({key0}).xls'
            df = pd.read_excel(fn)

    candidate_row = df.iloc[1]
    candidate_indexs = [idx for idx in range(len(candidate_row)) if type(candidate_row.iloc[idx]) is str]
    candidates = [remove_chars(candidate_row.iloc[idx], '\n') for idx in candidate_indexs]

    candidate_indexs += [-7, -5]
    candidates += ['總統_無效票', '總統_已領票未投票']

    ret_df = pd.DataFrame(columns=candidates)

    key1 = None
    key2 = None
    key3 = None
    for idx in range(5, len(df)):
        row = df.iloc[idx]
        x = row.iloc[0]
        y = row.iloc[1]
        z = row.iloc[2]

        votes = [str_to_int(row.iloc[cidx]) for cidx in candidate_indexs]

        if type(x) is str and len(remove_chars(x, '\u3000')) != 0:
            key1 = remove_chars(x, '\u3000')
            continue

        key2 = y
        key3 = z

        key = f'{key0}_{key1}_{key2}_{key3}'
        ret_df.loc[key] = votes

    return ret_df


def parse(year):
    key0_list = [
        '臺北市',
        '新北市',
        '桃園市',
        '臺中市',
        '臺南市',
        '高雄市',
        '新竹縣',
        '苗栗縣',
        '彰化縣',
        '南投縣',
        '雲林縣',
        '嘉義縣',
        '屏東縣',
        '宜蘭縣',
        '花蓮縣',
        '臺東縣',
        '澎湖縣',
        '金門縣',
        '連江縣',
        '基隆市',
        '新竹市',
        '嘉義市']

    pre_dfs = []
    for key0 in key0_list:
        pre_dfs.append(parse_president(year, key0))

    leg_dfs = []
    for key0 in key0_list:
        leg_dfs.append(parse_leg(year, key0))

    return pd.concat(pre_dfs), pd.concat(leg_dfs)


# analyze
def analyze_corr(pre_df: pd.DataFrame, leg_df: pd.DataFrame):
    pre_ratio_df = pre_df.div(pre_df.sum(axis='columns'), axis='index')
    leg_ratio_df = leg_df.div(leg_df.sum(axis='columns'), axis='index')

    df = pd.DataFrame()
    df[pre_ratio_df.columns] = pre_ratio_df
    df[leg_ratio_df.columns] = leg_ratio_df
    print(df.corr().to_csv())


def analyze_linear_regression(pre_df: pd.DataFrame, leg_df: pd.DataFrame):
    model = LinearRegression(fit_intercept=False)
    model.fit(leg_df, pre_df.iloc[:, :-2])
    y_pred = model.predict(leg_df)
    r2 = r2_score(pre_df.iloc[:, :-2], y_pred)
    coef_df = pd.DataFrame(model.coef_, index=pre_df.columns[:-2], columns=leg_df.columns)
    print(f'r2={r2}')
    print(coef_df.to_csv())


def analyze(pre_df: pd.DataFrame, leg_df: pd.DataFrame, consider_ratio=False):
    if consider_ratio:
        pre_df = pre_df.div(pre_df.sum(axis='columns'), axis='index')
        leg_df = leg_df.div(leg_df.sum(axis='columns'), axis='index')

    # x_df = leg_df
    # y_df = pre_df

    x_df = pre_df
    y_df = leg_df

    xnames = x_df.columns
    ynames = y_df.columns

    X = x_df.to_numpy()
    Y = y_df.to_numpy()
    B = cp.Variable((X.shape[1], Y.shape[1]))

    objective = cp.Minimize(cp.sum(cp.abs(X @ B - Y)))
    sum_constraints = [cp.sum(B[i, :]) == 1 for i in range(B.shape[0])]
    constraints = [
        B >= 0,
        B <= 1.
    ]
    problem = cp.Problem(objective, constraints + sum_constraints)
    problem.solve(verbose=True)

    optimized_coefficients = pd.DataFrame(B.value, columns=ynames, index=xnames)

    r2_values = []
    for i in range(Y.shape[1]):
        y_pred = X @ optimized_coefficients.iloc[:, i]
        r2 = r2_score(Y[:, i], y_pred)
        r2_values.append(r2)

    print(f'r2_values={r2_values}')
    print(optimized_coefficients.to_markdown())
    print(optimized_coefficients.to_csv())


def check_leading_dist(df: pd.DataFrame, filter=10):
    nums = df.to_numpy().flatten()
    cnts = [0] * 10
    for n in nums:
        idx = int(str(n)[0])
        cnts[idx] += (n > filter)

    s = sum(cnts)
    print(f'cnt: {s}')
    for i in range(1, 10):
        print(f'{i}: {cnts[i] / s}')


if __name__ == '__main__':
    pre_df_2024, leg_df_2024 = parse(2024)
    pre_df_2020, leg_df_2020 = parse(2020)
    pre_df_2016, leg_df_2016 = parse(2016)
