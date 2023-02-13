
#           BG-NBD ve Gamma-Gamma ile CLTV Tahmini

# İş Problemi

# İngiltere merkezli perakende şirketi satış ve pazarlama
# faaliyetleri için roadmap belirlemek istemektedir. Şirketin
# orta uzun vadeli plan yapabilmesi için var olan müşterilerin
# gelecekte şirkete sağlayacakları potansiyel değerin
# tahmin edilmesi gerekmektedir.


# Veri Seti Hikayesi

# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri
# arasındaki online satış işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu
# müşterisinin toptancı olduğu bilgisi mevcuttur.

# 8 Değişken 541.909 Gözlem
# InvoiceNo   Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
# StockCode   Ürün kodu ( Her bir ürün için eşsiz )
# Description Ürün ismi
# Quantity    Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate Fatura tarihi
# UnitPrice   Fatura fiyatı ( Sterlin )
# CustomerID  Eşsiz müşteri numarası
# Country     Ülke ismi



###############################################
#       Görev 1: BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması
###############################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Adım 1: 2010-2011 yıllarındaki veriyi kullanarak İngiltere’deki müşteriler için 6 aylık CLTV tahmini yapınız.

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

df.isnull().sum()
df.dropna(inplace=True)

df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df.describe().T
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

df= df[df["Country"]=="United Kingdom"]
df["Country"].shape

df["InvoiceDate"].max() #2011-12-09
today_date = dt.datetime(2011, 12, 11)

# cltv veri yapısının olusturulması

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days/7,
                                                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days/7], #musterinin yaşı=bugunun tarihinden ilk alısveris yaptıgı tarihi cıkar
                                         'Invoice': lambda Invoice: Invoice.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_df.head()
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency_weekly', 'T_weekly', 'frequency', 'monetary']
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] #işlem basına ortalama kazanc hesaplama

# cltv_df[cltv_df['frequency']==1]  olan 1350 satır var
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df.describe().T

# BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_weekly'],
        cltv_df['T_weekly'])

cltv_df["exp_sales_6_month"] = bgf.predict(4*6, #6 aylık
                                       cltv_df['frequency'],
                                       cltv_df['recency_weekly'],
                                       cltv_df['T_weekly'])

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],         #toplam işlem sayısı
                                        cltv_df['monetary'])

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv
# cltv_df= cltv_df.reset_index()
# cltv_final = df.merge(cltv_df, on="Customer ID", how="left")
cltv_df.sort_values("cltv",ascending=False)

# Adım 2: Elde ettiğiniz sonuçları yorumlayıp, değerlendiriniz.




###################################################
#    Görev 2: Farklı Zaman Periyotlarından Oluşan CLTV Analizi
###################################################

# Adım 1: 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.

cltv_1_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv_1_month"] = cltv_1_month

cltv_12_months = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv_12_months"] = cltv_12_months

# Adım 2: 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.

cltv_df.sort_values("cltv_1_month", ascending=False).head(10)
cltv_df.sort_values("cltv_12_months", ascending=False).head(10)

# Adım 3: Fark var mı? Varsa sizce neden olabilir?

# DİKKAT!
# Sıfırdan model kurulmasına gerek yoktur. Önceki görevde oluşturulan model üzerinden ilerlenebilir.



#############################################################################
#     Görev 3: Segmentasyon ve Aksiyon Önerileri
#############################################################################

# Adım 1: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız
# ve grup isimlerini veri setine ekleyiniz.

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])



# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("segment").agg({"cltv": ["min", "max", "mean", "sum"]})



