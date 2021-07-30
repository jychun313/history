class DataGovApi:

    def __init__(self, web_api, **dict_params):
        self.web_api = web_api
        self.dict_params = dict_params

    def get_table_items_apt_code(self):
        from bs4 import BeautifulSoup
        import requests

        lst = []
        for items in self.dict_params.items():
            items = "{}={}".format(items[0], items[1])
            lst.append(items)
            del items
        url = "{}?{}".format(self.web_api, "&".join(lst))
        del lst

        bs_obj = BeautifulSoup(requests.get(url).content, "html.parser")
        table_items = bs_obj.find('items')

        del bs_obj

        return table_items

    def get_table_items_address(self):
        from bs4 import BeautifulSoup
        import requests

        lst = []
        for items in self.dict_params.items():
            items = "{}={}".format(items[0], items[1])
            lst.append(items)
            del items
        url = "{}?{}".format(self.web_api, "&".join(lst))
        del lst

        bs_obj = BeautifulSoup(requests.get(url).content, "html.parser")
        if str(bs_obj.find("errmsg")) in str(bs_obj):
            DataGovApi.get_table_items_address(self)
        else:
            table_items = bs_obj.find('item')
            result_string = ""
            for i in ['bjdcode', 'dorojuso', 'kaptcode', 'kaptname', 'kaptaddr', 'kaptdacnt']:
                result_string += str(table_items.find(i))
                del i

            del bs_obj

            return result_string


class KakaoAPI:
    
    def __init__(self, headers):
        self.url = "https://dapi.kakao.com/v2/local/search/keyword"
        self.headers = headers # format {"authorization": "AUTH KEY"}
    
    def library(self, path_library=None):
        import pandas as pd
        import requests

        df_info_library = pd.read_excel("{}전국도서관표준데이터.xls".format(path_library))
        df_info_library.insert(0, "도서관명_no_space_lower", df_info_library["도서관명"].str.replace(" ", "").str.lower())
        df_info_library = df_info_library.drop(
            df_info_library[(df_info_library["시도명"] == "서울특별시") & (df_info_library["시군구명"] == "관악구") &
                            (df_info_library["도서관명_no_space_lower"] == "조원작은도서관")].index, axis=0).reset_index(drop=True)

        df = pd.DataFrame(columns=['place_name', 'address_name', 'road_address_name', 'y', 'x'])
        for ind in df_info_library.index:
            name_one = df_info_library.loc[ind, "도서관명_no_space_lower"]
            name_two = df_info_library.loc[ind, "도서관명"]

            r = requests.get(self.url, headers=self.headers, params={"query": "{}".format(name_one)}).json()

            if len(r.get("documents")) != 0:
                for l in r.get("documents"):
                    df = df.append(
                        pd.DataFrame(data=[l])[["place_name", "address_name", "road_address_name", "y", "x"]],
                        sort=False, ignore_index=True)
                    del l
            else:
                r = requests.get(self.url, headers=self.headers, params={"query": "{}".format(name_two)}).json()
                for l in r.get("documents"):
                    df = df.append(
                        pd.DataFrame(data=[l])[["place_name", "address_name", "road_address_name", "y", "x"]],
                        sort=False, ignore_index=True)
                    del l
            del name_one, name_two, r
        
        return df
    
    def park(self, path_park=None):
        import pandas as pd
        import requests
        import numpy as np

        lst_city_name = ["서울특별시", "인천광역시", "경기도", "충청남도", "충청북도", "대전광역시", "대구광역시"]
        map_city_name = {"강원도": "강원",
                         "경기도": "경기",
                         "경상남도": "경남",
                         "경상북도": "경북",
                         "광주광역시": "광주",
                         "대구광역시": "대구",
                         "대전광역시": "대전",
                         "부산광역시": "부산",
                         "서울특별시": "서울",
                         "울산광역시": "울산",
                         "인천광역시": "인천",
                         "전라남도": "전남",
                         "전라북도": "전북",
                         "제주특별자치도": "제주",
                         "충청남도": "충남",
                         "충청북도": "충북"}

        df_info_park = pd.read_excel("{}전국도시공원정보표준데이터-20200508.xlsx".format(path_park))

        if "관리번호" in df_info_park.loc[0, :].tolist():
            df_info_park.columns = df_info_park.loc[0, :].tolist()
            df_info_park = df_info_park.drop(0, axis=0).reset_index(drop=True)
        df_info_park = df_info_park[["공원명", "공원구분", "위도", "경도", "제공기관명", "소재지도로명주소", "소재지지번주소"]]
        df_info_park = df_info_park[df_info_park["제공기관명"] != "세종특별자치시"].reset_index(drop=True)
        df_info_park.insert(0, "제공기관명_extract",
                            df_info_park["제공기관명"].str.extract("({})".format("|".join(map_city_name.keys()))))

        df_info_park_isnull = df_info_park[df_info_park["제공기관명_extract"].isnull()].copy()
        df_info_park_isnull["제공기관명_extract"] = df_info_park_isnull["소재지도로명주소"].str.split(" ", expand=True)[0]
        df_info_park_notnull = df_info_park[df_info_park["제공기관명_extract"].notnull()].copy()

        df_info_park = pd.concat([df_info_park_isnull, df_info_park_notnull], axis=0).sort_index()
        del df_info_park_isnull, df_info_park_notnull

        for col in ["공원명_kakao_api", "주소_kakao_api", "위도__kakao_api", "경도_kakao_api"]:
            df_info_park[col] = np.nan
            del col

        for ind in df_info_park.index:
            # 소재지도로주소
            dorojuso = df_info_park.loc[ind, "소재지도로명주소"]
            # 소재지지번주소
            jibun = df_info_park.loc[ind, "소재지지번주소"]

            r = requests.get(self.url, headers=self.headers, params={"query": "{}".format(dorojuso)}).json()

            if len(r.get("documents")) != 0:
                for l in r.get("documents"):
                    df_info_park.loc[ind, "공원명_kakao_api"] = l["place_name"]
                    df_info_park.loc[ind, "주소_kakao_api"] = l["address_name"]
                    df_info_park.loc[ind, "위도__kakao_api"] = l["y"]
                    df_info_park.loc[ind, "경도_kakao_api"] = l["x"]
                    del l
            else:
                if jibun is np.nan:
                    continue
                else:
                    r = requests.get(self.url, headers=self.headers, params={"query": "{}".format(jibun)}).json()
                    for l in r.get("documents"):
                        df_info_park.loc[ind, "공원명_kakao_api"] = l["place_name"]
                        df_info_park.loc[ind, "주소_kakao_api"] = l["address_name"]
                        df_info_park.loc[ind, "위도__kakao_api"] = l["y"]
                        df_info_park.loc[ind, "경도_kakao_api"] = l["x"]
                        del l
            del dorojuso, jibun, r

        if df_info_park[df_info_park["위도__kakao_api"].notnull()].shape[0] != df_info_park[df_info_park["경도_kakao_api"].notnull()].shape[0]:
            raise ValueError("Need to check 위도, 경도..")
        else:
            return df_info_park
    
    def school(self, path_school=None):
        import pandas as pd
        import requests
        import numpy as np
        from functools import reduce

        map_city_name = {"강원도": "강원",
                         "경기도": "경기",
                         "경상남도": "경남",
                         "경상북도": "경북",
                         "광주광역시": "광주",
                         "대구광역시": "대구",
                         "대전광역시": "대전",
                         "부산광역시": "부산",
                         "서울특별시": "서울",
                         "울산광역시": "울산",
                         "인천광역시": "인천",
                         "전라남도": "전남",
                         "전라북도": "전북",
                         "제주특별자치도": "제주",
                         "충청남도": "충남",
                         "충청북도": "충북"}

        year = 2019
        file_name_elementary = ""
        file_name_middle = ""
        if year == 2019:
            file_name_elementary += "2019년도_학교기본정보_초등학교"
            file_name_middle += "2019년도_학교기본정보_중학교"
        elif year == 2020:
            file_name_elementary += "학교기본정보(초)"
            file_name_middle += "학교기본정보(중)"

        df_school_map_elementary = pd.DataFrame()
        df_school_map_middle = pd.DataFrame()

        if year == 2019:
            temp_elementary = pd.read_csv("{}{}.csv".format(path_school, file_name_elementary))
            del file_name_elementary
            temp_elementary.columns = [x.replace(" \n", "") for x in temp_elementary.columns]
            df_school_map_elementary = df_school_map_elementary.append(temp_elementary, sort=False, ignore_index=True)
            del temp_elementary

            temp_middle = pd.read_csv("{}{}.csv".format(path_school, file_name_middle))
            del file_name_middle
            temp_middle.columns = [x.replace(" \n", "") for x in temp_middle.columns]
            df_school_map_middle = df_school_map_middle.append(temp_middle, sort=False, ignore_index=True)
            del temp_middle
        elif year == 2020:
            temp_elementary = pd.read_excel("{}학교기본정보(초).xlsx".format(path_school))
            temp_elementary.columns = [x.replace(" \n", "") for x in temp_elementary.columns]
            df_school_map_elementary = df_school_map_elementary.append(temp_elementary, sort=False, ignore_index=True)
            del temp_elementary

            temp_middle = pd.read_excel("{}학교기본정보(중).xlsx".format(path_school))
            temp_middle.columns = [x.replace(" \n", "") for x in temp_middle.columns]
            df_school_map_middle = df_school_map_middle.append(temp_middle, sort=False, ignore_index=True)
            del temp_middle

        for col in ["학교명_kakao_api", "학교도로명주소_kakao_api", "위도_kakao_api", "경도_kakao_api"]:
            df_school_map_elementary[col] = np.nan
            df_school_map_middle[col] = np.nan
            del col

        for dfs in [df_school_map_elementary, df_school_map_middle]:
            for col in ["주소내역", "학교도로명 주소"]:
                temp = dfs[col].str.split(" ", expand=True)
                temp[0] = temp[0].map(map_city_name)
                temp = temp.fillna("")
                for ind in temp.index:
                    dfs.loc[ind, col] = reduce(lambda x, y: x + y,
                                               "{}".format(" ".join(temp.loc[ind, :].tolist()).strip()))
                    del ind
                del temp, col
            del dfs
        ########################################################################################################################
        for ind in df_school_map_elementary.index:
            school_name = df_school_map_elementary.loc[ind, "학교명"]
            address_name = df_school_map_elementary.loc[ind, "주소내역"]
            address_name = address_name.replace(" ", "")

            r = requests.get(self.url, headers=self.headers, params={"query": "{}".format(school_name)}).json()

            if len(r.get("documents")) != 0:
                for l in r.get("documents"):
                    if school_name in l["place_name"].replace(" ", "") and address_name in l["address_name"].replace(" ", ""):
                        df_school_map_elementary.loc[ind, "학교명_kakao_api"] = l["place_name"]
                        df_school_map_elementary.loc[ind, "학교도로명주소_kakao_api"] = l["address_name"]
                        df_school_map_elementary.loc[ind, "위도_kakao_api"] = l["y"]
                        df_school_map_elementary.loc[ind, "경도_kakao_api"] = l["x"]
                        del l
                        break
            del school_name, address_name, r

        for ind in df_school_map_middle.index:
            school_name = df_school_map_middle.loc[ind, "학교명"]
            address_name = df_school_map_middle.loc[ind, "주소내역"]
            address_name = address_name.replace(" ", "")

            r = requests.get(self.url, headers=self.headers, params={"query": "{}".format(school_name)}).json()

            if len(r.get("documents")) != 0:
                for l in r.get("documents"):
                    if school_name in l["place_name"].replace(" ", "") and address_name in l["address_name"].replace(" ", ""):
                        df_school_map_middle.loc[ind, "학교명_kakao_api"] = l["place_name"]
                        df_school_map_middle.loc[ind, "학교도로명주소_kakao_api"] = l["road_address_name"]
                        df_school_map_middle.loc[ind, "위도_kakao_api"] = l["y"]
                        df_school_map_middle.loc[ind, "경도_kakao_api"] = l["x"]
                        del l
                        break
            del school_name, r
        
        if df_school_map_elementary[df_school_map_elementary["위도_kakao_api"].notnull()].shape[0] != \
                df_school_map_elementary[df_school_map_elementary["경도_kakao_api"].notnull()].shape[0]:
            raise ValueError("초등학교) Need to check 위도, 경도..")
        if df_school_map_middle[df_school_map_middle["위도_kakao_api"].notnull()].shape[0] != \
                df_school_map_middle[df_school_map_middle["경도_kakao_api"].notnull()].shape[0]:
            raise ValueError("중학교) Need to check 위도, 경도..")
        return df_school_map_elementary, df_school_map_middle
    
    def government_office(self, path_government_office=None):
        import pandas as pd
        import requests

        df_info_government_office = pd.read_csv("{}전국공공시설개방정보표준데이터.csv".format(path_government_office),
                                                encoding="ms949")
        df_info_government_office = df_info_government_office[df_info_government_office["제공기관명"].notnull()].reset_index(
            drop=True)
        df_info_government_office["개방장소명_extract"] = df_info_government_office["개방장소명"].str.extract("(주민센터|구청)")
        df_info_government_office = df_info_government_office[
            df_info_government_office["개방장소명_extract"].notnull()].reset_index(drop=True).drop(["개방장소명_extract"], axis=1)
        df_info_government_office = df_info_government_office[
            ['개방장소명', '소재지도로명주소', '관리기관명', '사용안내전화번호', '홈페이지주소', '위도', '경도']].copy()

        df = pd.DataFrame(columns=['address_name', 'category_group_code', 'category_group_name',
                                          'category_name', 'distance', 'id', 'phone', 'place_name', 'place_url',
                                          'road_address_name', 'x', 'y'])
        
        for ind in df_info_government_office.index:
            name = df_info_government_office.loc[ind, "개방장소명"]

            r = requests.get(self.url, headers=self.headers, params={"query": "{}".format(name)}).json()

            if len(r.get("documents")) != 0:
                for l in r.get("documents"):
                    df = df.append(pd.DataFrame(data=[l]), sort=False, ignore_index=True)
                    del l
            del name, r
        
        return df