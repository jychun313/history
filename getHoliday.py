def getHoliday(year=None, operationName=None, serviceKey=None):
    from bs4 import BeautifulSoup
    import pandas as pd
    import requests

    url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService'

    queryParams = "{}/{}?solYear={}&ServiceKey={}".format(url, operationName, year, serviceKey)
    url = queryParams

    result = requests.get(url)
    bs_obj = BeautifulSoup(result.content, "html.parser")

    col_df = ["datename", "locdate"]
    df = pd.DataFrame(columns=col_df)
    for col in col_df:
        for val in bs_obj.select(col):
            val = str(val)
            val = val.replace("<{}>".format(col), "").replace("</{}>".format(col), "")
            df = df.append(pd.DataFrame(data=[val], columns=[col]), ignore_index=True, sort=False)
            del val
        del col
    del col_df

    df_datename = df[["datename"]].copy().dropna(axis=0).reset_index(drop=True)
    df_locate = df[["locdate"]].copy().dropna(axis=0).reset_index(drop=True)
    df = pd.concat([df_locate, df_datename], axis=1, ignore_index=True).rename(columns={0: "locdate", 1: "datename"})
    del df_datename, df_locate
    return df


def getHoliday_dataGoKr(path_output=None):
    """
    참고 문서 : https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15012690
    오퍼레이션명(영문) : 오퍼레이션명(국문)
    getHoliDeInfo : 국경일 정보조회
    getRestDeInfo : 공휴일 정보조회 >>>>
    getAnniversaryInfo : 기념일 정보조회
    get24DivisionsInfo : 24절기 정보조회
    getSundryDayInfo : 잡절 정보조회
    """
    import pandas as pd

    df = pd.DataFrame()
    for y in range(2021, 2030):
        getRestDeInfo = getHoliday(year=y, operationName="getRestDeInfo")
        getHoliDeInfo = getHoliday(year=y, operationName="getHoliDeInfo")
        df = df.append(getRestDeInfo, ignore_index=True, sort=False)
        df = df.append(getHoliDeInfo, ignore_index=True, sort=False)
        df.drop_duplicates(keep="first", ignore_index=True)
        del y, getRestDeInfo, getHoliDeInfo
    df.drop_duplicates(keep="first", inplace=True)
    return df


def getHoliday_fromModule(year_start=None, year_end=None, path_output=None):
    """
    example)
    year_start = int
    year_end = int
    """
    import holidays
    import pandas as pd
    import numpy as np

    kor_holidays = holidays.KOR()

    for v in range(year_start, year_end+1):
        dt = "{}-01-01".format(v)
        del v, dt

    df = pd.DataFrame()
    for k in kor_holidays:
        df = df.append(pd.DataFrame(data=[[k, kor_holidays[k]]], columns=["DATE", "DAY"]),
                       ignore_index=True, sort=False)
        del k
    df["DATE"] = df["DATE"].astype(str)
    df = df[df["DATE"] >= "2019-07-01"].reset_index(drop=True)
    df.drop_duplicates(keep="first", inplace=True)

    year_start = df["DATE"].min().split("-")[0]
    year_end = df["DATE"].max().split("-")[0]
    lst_dt = df["DATE"].tolist()
    for dt in pd.date_range(start="{}-01-01".format(year_start), end="{}-12-31".format(year_end)):
        dt = dt.strftime("%Y-%m-%d")
        if dt not in lst_dt:
            df = df.append(pd.DataFrame(data=[dt], columns=["DATE"]), ignore_index=True, sort=False)
        del dt
    del year_start, year_end
    df.sort_values(by=["DATE"], ignore_index=True, inplace=True)
    df["HOLIDAY"] = np.nan

    lst_ind_holiday = df[df["DAY"].notnull()].index.tolist()
    lst_ind_notholiday = df[df["DAY"].isnull()].index.tolist()

    df.loc[lst_ind_holiday, "HOLIDAY"] = "Y"
    df.loc[lst_ind_notholiday, "HOLIDAY"] = "N"
    del lst_ind_holiday, lst_ind_notholiday
    df.drop(["DAY"], axis=1, inplace=True)
    df.insert(1, "DAY", pd.to_datetime(df["DATE"]).apply(lambda x: x.strftime("%A")))
    df.insert(2, "WEEKENDS", df["DAY"].copy())
    map_weekends ={"Monday": "N", "Tuesday": "N", "Wednesday": "N", "Thursday": "N", "Friday": "N", "Saturday": "Y", "Sunday": "Y"}
    df["WEEKENDS"] = df["WEEKENDS"].map(map_weekends)
    df["WEEKENDSHOLIDAY"] = df["WEEKENDS"] + df["HOLIDAY"]
    return df

