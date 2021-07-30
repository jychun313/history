"""
https://plotly.com/python/subplots/
- https://plotly.com/python/marker-style/
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
import pickle
import gzip
from scipy.optimize import curve_fit
from glob import glob
import cvxpy as cp
import locale
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 100)
locale.setlocale(locale.LC_ALL, '')

path_export_image = "path"
path_export_excel = "path"
path_agency_output = "path"
path_agency_output = sorted(Path(path_agency_output).iterdir(), key=os.path.getmtime)


# 로그함수 추세선 예측
def log_func(x=None, a=None, b=None):
    return a*np.log(x+1)+b


class VariableList(list):
    def search(self, name):
        '''Return all contacts that contain the search value in their name.'''
        matching_contacts = []
        for contact in self:
            if name in contact.name:
                matching_contacts.append(contact)
        return matching_contacts


class BudgetOptimization:
    all_variables = VariableList()

    def __init__(self, channel, b1, b2):
        import cvxpy as cp
        self.channel = channel
        self.x = cp.Variable(pos=True)
        self.b1 = b1
        self.b2 = b2
        self.equation = self.b1 * cp.log(self.x + 1) + self.b2
        self.all_variables.append(self)


def ad_cost():
    p = "path"
    df_ad_cost = pd.DataFrame()
    for f in sorted(glob("{}*".format(p))):
        df_ad_cost = df_ad_cost.append(pd.read_excel(f), sort=False, ignore_index=True)
        del f
    del p

    df_ad_cost = df_ad_cost[["Date", "채널", "adCost"]]
    df_ad_cost["채널"] = df_ad_cost["채널"].str.replace(" ", "")
    df_ad_cost = df_ad_cost[df_ad_cost["adCost"] != 0]

    return df_ad_cost


def read_file_date(max_date=None, path_import=None):
    file_name_income_base_with_cat_adcost = "df_income_base_g_"

    file_nm_excel_date = "df_lst_used_income_base.csv"

    print("Importing {}{}".format(path_import, file_nm_excel_date))
    df_lst_excel_date = pd.read_csv("{}{}".format(path_import, file_nm_excel_date))
    del path_import, file_nm_excel_date
    df_lst_excel_date["FILE_DATE"] = df_lst_excel_date["FILE"].apply(lambda x: "_".join(x.split("_")[-3:-1]))
    df_lst_excel_date_final = df_lst_excel_date[df_lst_excel_date["NO"] == "마감"].copy().reset_index(drop=True)
    df_lst_excel_date_final = df_lst_excel_date_final.append(
        df_lst_excel_date[~df_lst_excel_date["FILE_DATE"].str.contains(
            "|".join(df_lst_excel_date_final["FILE_DATE"]))].drop_duplicates(["FILE_DATE"], keep="last"),
        ignore_index=True, sort=False)
    del df_lst_excel_date
    df_lst_excel_date_final["FILE_DATE"] += "_"
    for i in range(1, 10):
        df_lst_excel_date_final["FILE_DATE"] = df_lst_excel_date_final["FILE_DATE"].str.replace("_{}_".format(i),
                                                                                                "_0{}_".format(i))
        del i
    df_lst_excel_date_final["FILE_DATE"] = df_lst_excel_date_final["FILE_DATE"].apply(lambda x: x[:-1])
    df_lst_excel_date_final["NO"] = df_lst_excel_date_final["NO"].map({"1차": "1cha", "2차": "2cha", "마감": "finish"})
    df_lst_excel_date_final["FILE_READ"] = file_name_income_base_with_cat_adcost + df_lst_excel_date_final[
        "FILE_DATE"] + "_" + df_lst_excel_date_final["NO"] + ".pickle"
    del file_name_income_base_with_cat_adcost
    df_lst_excel_date_final.drop(["NO", "FILE_DATE"], axis=1, inplace=True)
    df_lst_excel_date_final["dt"] = df_lst_excel_date_final["FILE_READ"].apply(
        lambda x: f'{x.split("_")[4]}-{x.split("_")[5]}')
    if max_date != None:
        df_lst_excel_date_final = df_lst_excel_date_final[df_lst_excel_date_final["dt"] <= max_date]
    df_lst_excel_date_final.drop(["dt"], axis=1, inplace=True)
    return df_lst_excel_date_final


def read_income_base(max_date=None, path_import=None):
    df_lst_excel_date_final = read_file_date(max_date=max_date)

    df_income_base_g = pd.DataFrame()
    for file in df_lst_excel_date_final["FILE_READ"].unique():
        with gzip.open("{}{}".format(path_import, file),
                       "rb") as f:
            temp = pickle.load(f)
        for col in [x for x in temp.columns if "DB" in x]:
            temp.rename(columns={col: col.split("_")[0]}, inplace=True)
            del col
        df_income_base_g = df_income_base_g.append(temp, sort=False, ignore_index=True)
        del file, temp
    for col in [x for x in df_income_base_g.columns if "adCost" in x]:
        df_income_base_g.drop(col, axis=1, inplace=True)
        del col
    df_income_base_g.drop(["PROMOTION_DATE"], axis=1, inplace=True)

    # 2019년 1월 ~ 2019년 6월 데이터
    with gzip.open("/../../file_name", "rb") as f:
        df_income_base = pickle.load(f)
        del f

    df_income_base_g = pd.concat([df_income_base, df_income_base_g], axis=0, ignore_index=True, sort=False)
    del df_income_base
    df_income_base_g = df_income_base_g[df_income_base_g["Date"].notnull()].reset_index(drop=True)
    return df_income_base_g, df_lst_excel_date_final


def run_BudgetOptimization(val=None, keep_data_points=None, today=datetime.today().strftime("%Y%m%d"), max_date=None):

    max_date_plus_one_month = f"{max_date}-{monthrange(year=int(max_date.split('-')[0]), month=int(max_date.split('-')[1]))[1]}"
    max_date_plus_one_month = (pd.to_datetime(max_date_plus_one_month) + timedelta(days=1)).strftime("%Y-%m")

    df_income_base_g, df_lst_excel_date_final = read_income_base(max_date=max_date)
    df_income_base_g["채널"] = df_income_base_g["채널"].str.replace(" ", "")

    df_ad_cost = ad_cost()
    df_ad_cost_target = df_ad_cost[df_ad_cost["Date"] == max_date_plus_one_month].copy().reset_index(drop=True)
    df_ad_cost = df_ad_cost[df_ad_cost["Date"] < max_date_plus_one_month]
    filtered = df_income_base_g["채널"].isin(df_ad_cost_target["채널"].unique().tolist())
    df_income_base_g = df_income_base_g[filtered].reset_index(drop=True)
    del filtered

    temp = df_income_base_g.copy()
    temp.insert(temp.columns.tolist().index("채널"), "채널_original", temp["채널"].copy())
    temp["채널"] = temp["채널"].str.replace(" ", "")
    df_mkt_merge_filter = pd.merge(temp, df_ad_cost, on=["Date", "채널"], how="outer").drop(["채널"], axis=1).rename(
        columns={"채널_original": "채널"})
    df_mkt_merge_filter = df_mkt_merge_filter[df_mkt_merge_filter["대상"].notnull()].reset_index(drop=True)
    if temp.shape[0] != df_mkt_merge_filter.shape[0]:
        raise ValueError("Need to check data..")
    else:
        del temp
    # # 내부 채널 포함 )
    # df_mkt_merge_filter["adCost"].fillna(0, inplace=True)
    # 광고비 사용한 채널만 )
    df_mkt_merge_filter = df_mkt_merge_filter[df_mkt_merge_filter["adCost"].notnull()].reset_index(drop=True)
    for col in [x for x in df_mkt_merge_filter.columns if "DB" in x]:
        df_mkt_merge_filter[col] = df_mkt_merge_filter[col].fillna(0).astype(int)
        del col
    df_mkt_merge_filter.sort_values(by=["채널", "Date"], ignore_index=True, inplace=True)
    df_mkt_merge_filter_g = df_mkt_merge_filter.copy()
    df_mkt_merge_filter_g["DATA POINTS"] = 1
    df_mkt_merge_filter_g = df_mkt_merge_filter_g.groupby(["채널"]).agg({"DATA POINTS": "sum", "Date": list}).reset_index().rename(columns={"Date": "기간"})

    lst_ind_channel_keep = df_mkt_merge_filter_g[df_mkt_merge_filter_g["DATA POINTS"] >= keep_data_points].index.tolist()
    lst_ind_channel_drop = df_mkt_merge_filter_g[df_mkt_merge_filter_g["DATA POINTS"] < keep_data_points].index.tolist()
    df_mkt_merge_filter_g.loc[lst_ind_channel_keep, "구분"] = "사용"
    df_mkt_merge_filter_g.loc[lst_ind_channel_drop, "구분"] = "미사용"
    del lst_ind_channel_keep, lst_ind_channel_drop
    df_mkt_merge_filter_g["기간"] = df_mkt_merge_filter_g["기간"].apply(lambda x: ", ".join(x))

    if val == "상담신청DB":
        f = "{}iscream_edu_data_points_{}_{}.csv".format(path_export_excel, max_date, today)
        print("Exporting {}".format(f))
        df_mkt_merge_filter_g.to_csv(f, encoding='utf-8-sig', index=False)
        del f

    df_mkt_merge_filter_g.drop(["DATA POINTS", "기간"], axis=1, inplace=True)
    df_mkt_merge_filter = pd.merge(df_mkt_merge_filter, df_mkt_merge_filter_g, on=["채널"], how="outer")
    del df_mkt_merge_filter_g
    df_mkt_merge_filter = df_mkt_merge_filter[(df_mkt_merge_filter["구분"] == "사용") &
                                              (df_mkt_merge_filter["소분류"] != "홈쇼핑")].reset_index(drop=True).drop(
        ["대상", "대분류", "소분류", "구분"], axis=1).rename(columns={"adCost": "광고비"})
    del keep_data_points

    # cvxpy 사용
    df = df_mkt_merge_filter.copy()
    data_list = []
    data_list_name = []
    for ch in df["채널"].unique():
        temp = df[df["채널"] == ch].copy().reset_index(drop=True)
        data_list.append(temp)
        data_list_name.append(ch)
        del ch, temp

    ## 데이터를 통해 추세선 계수값 예측
    data_list_final = []
    data_list_name_final = []
    opt_cov_list = []

    for i in range(0, len(data_list)):
        lst = []
        # array 길이가 1인것은 log_func 계산이 안되므로 제외 필요..
        if len(data_list[i]['광고비'].tolist()) == 1:
            continue
        a, b = curve_fit(log_func, data_list[i]['광고비'].tolist(), data_list[i][val].tolist(), maxfev=5000)
        """
        Notes: maxfev is the maximum number of func evaluations tried; you
        can try increasing this value if the fit fails.
        If the program returns a good chi-squared but an infinite
        covariance and no parameter uncertainties, it may be
        because you have a redundant parameter;
        try fitting with a simpler function.
        """
        data_list_final.append(data_list[i])
        data_list_name_final.append(data_list_name[i])
        lst.append(data_list_name[i])
        lst.append(a)
        lst.append(b)
        opt_cov_list.append(lst)
        del a, b, lst, i

    # 각 채널 당 기울기값 부여 (b1_list),  각 채널 당 절편값 부여 (b2_list)
    b1_list = []
    b2_list = []
    for i in range(len(opt_cov_list)):
        temp = opt_cov_list[i][1]
        b1_list.append(temp[0])
        b2_list.append(temp[1])
        del i, temp
    value_list = ["Total"] + data_list_name_final
    value_table = pd.DataFrame(value_list, columns=['Channels'])
    value_table.style.set_properties(**{'text-align': 'right'})
    del value_list

    budget_optimization = BudgetOptimization(data_list_name_final[0], b1_list[0], b2_list[0])
    for i in range(1, len(data_list_name_final)):
        budget_optimization = BudgetOptimization(data_list_name_final[i], b1_list[i], b2_list[i])
        del i

    """
    Expression(%s, %s, %s)" % (self.curvature, self.sign, self.shape)
    curvature == CONCAVE (O)
    curvature == CONVEX (X)
    """
    df_string_omit = pd.DataFrame()
    lst_channel = []
    lst_x = []
    lst_b1 = []
    lst_b2 = []
    lst_equation = []
    for i in range(len(data_list_name_final)):
        channel = budget_optimization.all_variables[i].channel
        x = budget_optimization.all_variables[i].x
        b1 = budget_optimization.all_variables[i].b1
        b2 = budget_optimization.all_variables[i].b2
        equation = budget_optimization.all_variables[i].equation
        curvature = budget_optimization.all_variables[i].equation.curvature
        if budget_optimization.all_variables[i].equation.curvature == "CONCAVE":
            lst_channel.append(channel)
            lst_x.append(x)
            lst_b1.append(b1)
            lst_b2.append(b2)
            lst_equation.append(equation)
        else:
            print("{} (channel) is not CONCAVE.. It is {}..".format(channel, curvature))
            df_string_omit = df_string_omit.append(pd.DataFrame(data=["{} (channel) is not CONCAVE.. It is {}..".format(
                channel, curvature)]), sort=False, ignore_index=True)
        del i, channel, x, b1, b2, equation, curvature

    ########################################################################################################################
    filtered = df_ad_cost_target["채널"].isin(lst_channel)
    df_ad_cost_target = df_ad_cost_target[filtered]
    if len(lst_channel) != df_ad_cost_target["채널"].nunique():
        raise ValueError("Need to check data..")
    else:
        del filtered
        total_budget = int(df_ad_cost_target["adCost"].sum())

    budget_list = [total_budget]
    for l in range(0, len(budget_list)):

        budget_name = str(round(budget_list[l]/100000000, 1))+"억 원"

        sale = sum(lst_equation)
        budget = sum(lst_x)
        constraints = [budget_list[l] >= budget] + [i >= 0 for i in lst_x] + [i >= 0 for i in lst_equation]

        obj_max_sale = cp.Maximize(sale)
        prob_max_sale = cp.Problem(obj_max_sale, constraints)
        prob_max_sale.solve(solver=cp.MOSEK, verbose=True)  # solver=cp.MOSEK / cp.SCS
        print("상태 :", prob_max_sale.status)

        if budget.value == None:
            continue
        value_list = []
        value_list.append(["Total", locale.format('%.2f', budget.value, 1), locale.format('%.2f', sale.value, 1)])
        for i in range(len(lst_equation)):
            value_list.append([lst_channel[i], locale.format('%.2f', lst_x[i].value, 1), locale.format('%.2f', lst_equation[i].value, 1)])
            del i

        value = pd.DataFrame(value_list, columns=['Channels', budget_name + " Budget", budget_name + " DB"])
        value.style.set_properties(**{'text-align': 'right'})
        value_table[budget_name+" Budget"] = value[budget_name+" Budget"]
        value_table[budget_name+" DB"] = value[budget_name+" DB"]
        del l, value, budget_name, sale, budget, constraints, obj_max_sale, prob_max_sale, value_list
    del b1_list, b2_list, budget_list
    value_table.insert(0, "DBTYPE", val)
    del data_list, data_list_name
    # Export files #########################################################################################################
    if val == "상담신청DB":
        df_lst_excel_date_final = df_lst_excel_date_final[["LEADING_DATE", "FILE"]].rename(
            columns={"LEADING_DATE": "인입월", "FILE": "파일명"})
        temp = pd.DataFrame(columns=["인입월", "파일명"])
        for m in range(1, 7):
            temp_search_date = "20190{}{}".format(m+2, monthrange(year=2019, month=m+2)[1])
            temp = temp.append(pd.DataFrame({"인입월": ["2019년 {}월 인입자".format(m)], "파일명": ["income_base_login_{}_2019_{}_8.xlsx".format(temp_search_date, m)]}),
                               sort=False, ignore_index=True)
            del m
        df_lst_excel_date_final = pd.concat([temp, df_lst_excel_date_final], axis=0, ignore_index=True)
        del temp
        file_nm_date_final = "{}iscream_edu_file_list_{}_{}.csv".format(path_export_excel, max_date, today)
        print("Exporting {}".format(file_nm_date_final))
        df_lst_excel_date_final.to_csv(file_nm_date_final, encoding='utf-8-sig', index=False)
        del file_nm_date_final

    file_name_ad_cost_target = "{}ad_cost_target_for_{}_{}_{}.csv".format(path_export_excel, val, max_date, today)
    df_ad_cost_target.to_csv(file_name_ad_cost_target, encoding='utf-8-sig', index=False)
    del df_ad_cost_target, file_name_ad_cost_target

    df.rename(columns={"Date": "기간"}, inplace=True)
    file_nm_merge_filter = "{}iscream_edu_data_{}_{}_{}.csv".format(path_export_excel, val, max_date, today)
    print("Exporting {}".format(file_nm_merge_filter))
    df.drop(["목표"], axis=1).to_csv(file_nm_merge_filter, encoding='utf-8-sig', index=False)
    del df, file_nm_merge_filter

    file_nm_export = "{}iscream_edu_optimization_result_small_channel_{}_{}_{}.csv".format(path_export_excel, val, max_date, today)
    print("Exporting {}".format(file_nm_export))
    value_table.to_csv(file_nm_export, encoding='utf-8-sig', index=False)
    del value_table, file_nm_export

    file_nm_string_omit = "{}omitted_channels_{}_{}_{}.txt".format(path_export_excel, val, max_date, today)
    df_string_omit.to_csv(file_nm_string_omit, header=False, index=False)
    del file_nm_string_omit, df_string_omit


def create_marketing_budget_optimization(max_date=None):
    today = datetime.today().strftime("%Y%m%d")

    file_nm_result_0 = "{}iscream_edu_optimization_result_small_channel_상담신청DB_{}_{}.csv".format(path_export_excel, max_date, today)
    file_nm_result_1 = "{}iscream_edu_optimization_result_small_channel_신규DB_{}_{}.csv".format(path_export_excel, max_date, today)
    file_nm_result_2 = "{}iscream_edu_file_list_{}_{}.csv".format(path_export_excel, max_date, today)
    file_nm_result_3 = "{}iscream_edu_data_상담신청DB_{}_{}.csv".format(path_export_excel, max_date, today)
    file_nm_result_4 = "{}iscream_edu_data_신규DB_{}_{}.csv".format(path_export_excel, max_date, today)
    file_nm_result_5 = "{}iscream_edu_data_points_{}_{}.csv".format(path_export_excel, max_date, today)

    if file_nm_result_0 in [x.replace("\\", "/") for x in glob("{}*".format(path_export_excel))] and \
            file_nm_result_1 in [x.replace("\\", "/") for x in glob("{}*".format(path_export_excel))] and \
            file_nm_result_2 in [x.replace("\\", "/") for x in glob("{}*".format(path_export_excel))] and \
            file_nm_result_3 in [x.replace("\\", "/") for x in glob("{}*".format(path_export_excel))] and \
            file_nm_result_4 in [x.replace("\\", "/") for x in glob("{}*".format(path_export_excel))] and \
            file_nm_result_5 in [x.replace("\\", "/") for x in glob("{}*".format(path_export_excel))]:

        print("Reading {}".format(file_nm_result_0))
        df_result_0 = pd.read_csv(file_nm_result_0)
        del file_nm_result_0

        for col in [x for x in df_result_0.columns if "Budget" in x or " DB" in x]:
            df_result_0.rename(columns={col: "{}_({})".format(col, df_result_0["DBTYPE"].unique()[0])}, inplace=True)
            del col
        df_result_0.drop(["DBTYPE"], axis=1, inplace=True)

        print("Reading {}".format(file_nm_result_1))
        df_result_1 = pd.read_csv(file_nm_result_1)
        del file_nm_result_1

        for col in [x for x in df_result_1.columns if "Budget" in x or " DB" in x]:
            df_result_1.rename(columns={col: "{}_({})".format(col, df_result_1["DBTYPE"].unique()[0])}, inplace=True)
            del col
        df_result_1.drop(["DBTYPE"], axis=1, inplace=True)

        df_result_0_1 = pd.merge(df_result_0, df_result_1, on=["Channels"], how="outer")
        for col in [x for x in df_result_0_1.columns if "Budget" in x]:
            lst_inds = df_result_0_1[df_result_0_1[col].notnull()].index.tolist()
            df_result_0_1.loc[lst_inds, col] = df_result_0_1.loc[lst_inds, col].apply(
                lambda x: "₩ {}".format(x))
            del lst_inds, col
        del df_result_0, df_result_1

        print("Reading {}".format(file_nm_result_2))
        df_result_2 = pd.read_csv(file_nm_result_2)
        del file_nm_result_2

        print("Reading {}".format(file_nm_result_3))
        df_result_3 = pd.read_csv(file_nm_result_3)
        del file_nm_result_3
        for col in [x for x in df_result_3.columns if "체험신청" in x or "DB" in x or "광고비" in x]:
            df_result_3[col] = df_result_3[col].apply(lambda x: "{0:,}".format(x))
            del col

        print("Reading {}".format(file_nm_result_4))
        df_result_4 = pd.read_csv(file_nm_result_4)
        del file_nm_result_4
        for col in [x for x in df_result_4.columns if "체험신청" in x or "DB" in x or "광고비" in x]:
            df_result_4[col] = df_result_4[col].apply(lambda x: "{0:,}".format(x))
            del col

        print("Reading {}".format(file_nm_result_5))
        df_result_5 = pd.read_csv(file_nm_result_5)
        del file_nm_result_5

        f = "{}marketing_budget_optimization_{}_{}.xlsx".format(path_export_excel, max_date, today)
        print("Exporting {}".format(f))
        with pd.ExcelWriter(f) as writer:
            df_result_0_1.to_excel(writer, sheet_name="모델 결과값", index=False)
            df_result_2.to_excel(writer, sheet_name="파일명 리스트", index=False)
            df_result_3.to_excel(writer, sheet_name="데이터(상담신청DB)", index=False)
            df_result_4.to_excel(writer, sheet_name="데이터(신규DB)", index=False)
            df_result_5.to_excel(writer, sheet_name="DATA POINTS", index=False)


# 데이터 검증 - 신규DB 대비 단가 차이 - 필요한 Date 만 필터한 후
def verification(dbtype=None, lst_dt=None):
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    today = datetime.today().strftime("%Y%m%d")

    for tup in lst_dt:
        print(f"Working on {tup}")
        df_real, _ = read_income_base(max_date=tup[1])
        df_real = df_real[df_real["Date"] == tup[1]]
        df_real["채널"] = df_real["채널"].str.replace(" ", "")
        df_real = df_real[["채널", "상담신청DB", "신규DB"]].rename(columns={"채널": "Channels", "상담신청DB": "상담신청DB_실제값", "신규DB": "신규DB_실제값"})

        df_pred = pd.read_excel("/../../marketing_budget_optimization_{}_{}.xlsx".format(tup[0], today), sheet_name="모델 결과값")
        df_pred = df_pred[df_pred["Channels"] != "Total"].reset_index(drop=True)
        for col in df_pred.columns[1:]:
            df_pred[col] = df_pred[col].str.replace(",", "").str.replace("₩ ", "")
            df_pred[col] = df_pred[col].astype(float)
            del col

        df_ad_cost = pd.read_csv("/../../ad_cost_target_for_{}_{}_{}.csv".format(dbtype, tup[0], today))

        df_merge = pd.merge(df_real, df_pred, on=["Channels"], how="inner")
        df_merge = df_merge[["Channels"] + [x for x in df_merge.columns if dbtype in x]]
        float_budget = float([x for x in df_pred.columns if dbtype in x][0].split("억")[0])
        for no in [float_budget]:
            col_nm = f"{dbtype[:2]}_단가예측값({no}억)"
            df_merge.insert(df_merge.columns.tolist().index(f"{no}억 원 DB_({dbtype})") + 1, col_nm,
                            df_merge[f"{no}억 원 Budget_({dbtype})"] / df_merge[f"{no}억 원 DB_({dbtype})"])
            df_merge.loc[df_merge[df_merge[col_nm] == np.inf].index.tolist(), col_nm] = np.nan
            df_merge.rename(columns={f"{no}억 원 Budget_({dbtype})": f"{dbtype[:2]}_예산예측값({no}억)",
                                     f"{no}억 원 DB_({dbtype})": f"{dbtype[:2]}_DB예측값({no}억)"}, inplace=True)
            del no, col_nm
        # how="outer" >> 광고비는 사용했지만, 인입이 없는 경우도 포함해야됨.
        df_merge = pd.merge(df_ad_cost.rename(columns={"채널": "Channels"}).drop(["Date"], axis=1), df_merge,
                            on=["Channels"], how="outer").rename(columns={"adCost": "예산"})
        if int(df_ad_cost["adCost"].sum()) != int(df_merge["예산"].sum()):
            raise ValueError("Need to check data after merging with adcost..")
        df_merge.insert(df_merge.columns.tolist().index(f"{dbtype}_실제값")+1, "단가_실제값",
                        df_merge["예산"] / df_merge[f"{dbtype}_실제값"])

        for no in [float_budget]:
            col_nm = f"단가차이({no}억)"
            df_merge.insert(df_merge.columns.tolist().index(f"{dbtype[:2]}_단가예측값({no}억)") + 1, col_nm,
                            df_merge["단가_실제값"] - df_merge[f"{dbtype[:2]}_단가예측값({no}억)"])
            del no, col_nm

        channels = df_merge["Channels"].tolist()

        today = datetime.today().strftime("%Y%m%d")

        y_diff = df_merge[f"단가차이({float_budget}억)"].tolist()

        table_data_total = df_merge.copy()
        string_total_budget = table_data_total["예산"].sum()
        for col in [x for x in table_data_total.columns if "예산" in x or "단가" in x]:
            ind_notnull = table_data_total[table_data_total[col].notnull()].index.tolist()
            table_data_total.loc[ind_notnull, col] = table_data_total.loc[ind_notnull, col].apply(lambda x: "₩ {:,.2f}".format(x))
            del col, ind_notnull
        for col in [x for x in table_data_total.columns if "DB" in x]:
            ind_notnull = table_data_total[table_data_total[col].notnull()].index.tolist()
            table_data_total.loc[ind_notnull, col] = table_data_total.loc[ind_notnull, col].apply(lambda x: "{:,.2f} 건".format(x))
            del col, ind_notnull
        table_data_total = [table_data_total.columns.tolist()] + table_data_total.values.tolist()
        fig_total = ff.create_table(table_data_total, height_constant=60)
        for i in range(len(fig_total.layout.annotations)):
            fig_total.layout.annotations[i].font.size = 10
            del i
        del table_data_total

        trace1_total = go.Bar(x=channels, y=y_diff, xaxis='x2', yaxis='y2', name=f"단가차이({float_budget}억)")

        fig_total.add_traces([trace1_total])
        del trace1_total, float_budget

        # initialize xaxis2 and yaxis2
        fig_total['layout']['xaxis2'] = {}
        fig_total['layout']['yaxis2'] = {}

        # Edit layout for subplots
        # [0, 1] = [left, right] = [top, bottom]
        # In this case, 'autorange': 'reversed'
        fig_total.layout.yaxis.update({'domain': [0, .75]}) # [0, .45] # table 크리 조정
        fig_total.layout.yaxis2.update({'domain': [0.85, 1]}) # [.6, 1] # bar graph 크기 조정

        # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
        fig_total.layout.yaxis2.update({'anchor': 'x2'})
        fig_total.layout.xaxis2.update({'anchor': 'y2'})
        fig_total.layout.yaxis2.update({'title': '차이 (단가)'})

        # Update the margins to add a title and see graph x-labels.
        fig_total.layout.margin.update({'t':75, 'l':50})
        fig_total.layout.update({'title': f'{dbtype} (총 예산: {"₩ {:,.2f}".format(round(string_total_budget, 2))})// + (실제값이 큼), - (예측값이 큼)'})

        # Update the height because adding a graph vertically will interact with
        # the plot height calculated for the table
        fig_total.layout.update({'height':1500, 'width':1800})
        # fig_total.show()
        path_string_one = f"/../../{dbtype}_{tup[0]}_{today}.html"
        path_string_two = f"/../../{dbtype}_{tup[0]}_{today}.html"
        print(f"Exporting {path_string_one}")
        fig_total.write_html(path_string_one)
        print(f"Exporting {path_string_two}")
        fig_total.write_html(path_string_two)
        del path_string_one, path_string_two, fig_total
