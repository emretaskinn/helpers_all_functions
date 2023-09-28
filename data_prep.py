import pandas as pd
import numpy as np
import seaborn as sns
import zipfile
import pickle
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

def readfile_fromzip(zippath):
    import zipfile
    import pandas as pd
    zf = zipfile.ZipFile(zippath)
    file_list = pd.DataFrame(zf.namelist())
    file_list.columns = ['Files']
    print(file_list)

    selection = input('Please type down index of the file you would like to read: ')
    converted_selection = int(selection)
    filename = file_list.loc[converted_selection]
    if '.csv' in filename.values[0]:
        df = pd.read_csv(zf.open(f'{filename.values[0]}'))
        return df
    else:
        xls = pd.ExcelFile(zf.open(f'{filename.values[0]}'))
        df1 = pd.read_excel(zf.open(f'{filename.values[0]}'), sheet_name=xls.sheet_names[0])
        df2 = pd.read_excel(zf.open(f'{filename.values[0]}'), sheet_name=xls.sheet_names[1])
        return df1,df2

def outlier_lof(df,n_neighb=20):
    """
    Local Outlier Factor Yöntemi ile Aykırı Gözlem Analizi (LOF)
    :param df:
    :return:
    """
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=n_neighb)
    lof.fit_predict(df)

    # Skor değerleri gelmiştir.
    df_scores = lof.negative_outlier_factor_

    # Eşik değeri belirlenilmiştir.
    threshold = np.sort(df_scores)[9]

    # Belirlenen eşik değer veri setine uyarlanarak aykırı gözlemlerden kurtulunmuş olundu.
    outlier = df_scores > threshold
    df = df[outlier]
    return df

def lof_scores(df):
    """
    Her bir gözlemin LOF Score unu hesaplar
    :param df:
    :return:
    """
    clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    clf.fit_predict(df)
    df_scores=clf.negative_outlier_factor_
    return df_scores

def lof_threshold(df, df_scores, threshold):
    """
    Lof Score lara bakılarak Anormal olan veri silinir.
    :param df:
    :param df_scores:
    :param threshold:
    :return:
    """
    not_outlier = df_scores >threshold
    value = df[df_scores == threshold]
    outliers = df[~not_outlier]
    res = outliers.to_records(index=False)
    res[:] = value.to_records(index=False)
    not_outlier_df = df[not_outlier]
    outliers = pd.DataFrame(res, index = df[~not_outlier].index)
    df_res = pd.concat([not_outlier_df, outliers], ignore_index = True)
    return df_res

def data_summary(dataframe):
    print("-------------------DATA SUMMARY---------\n")
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("\n-------------------DATA SUMMARY END---------\n")

def num_summary(dataframe, numerical_col, plot=False):
    """
    Bir sayısal değişkenin çeyreklik değerlerini gösterir ve histogram olusturur

    example:
        for col in age_cols:
            num_summary(df, col, plot=True)
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("\n-------------NUM SUMMARY START--------------\n")
    print(dataframe[numerical_col].describe(quantiles).T)
    print("\n-------------NUM SUMMARY END--------------\n")
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def cat_summary(dataframe, col_name, plot=False):
    """
    Verilen kategorik değişken için frekans-oran detaylarını yazdırır.
    :param dataframe:
    :param col_name:
    :param plot:

    example:
        for col in cat_cols:
            cat_summary(df, col)

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def missing_values_table(dataframe, na_name=False):
    import pandas as pd
    import numpy as np
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    import pandas as pd
    import numpy as np
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def label_encoder(dataframe, binary_col):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    import pandas as pd
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    import pandas as pd
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    import numpy as np
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    kategorik değişkenlere göre hedef değişkenin ortalamasını verir

    example:
    for col in cat_cols:
        target_summary_with_cat(df,"SalePrice",col)
    """
    import pandas as pd
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def num_summary_with_target(dataframe, target, numeric_col):
    """
    hedef değişkene göre numerik değişkenlerin ortalamasını verir

    example:
    for col in cat_cols:
        target_summary_with_cat(df,"SalePrice",col)
    """
    print("-------------NUM SUMMARY WITH "+target+ " START------------\n")
    print(dataframe.groupby(target)[numeric_col].mean())
    print("\n-------------NUM SUMMARY WITH " + target + " END------------")

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    import numpy as np
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def standardization(df, column_name, min=1, max=5, time="No"):
    """
        Task        :   Dataframe içerisinde verilen bir alan için verilen aralıklarda STANDARTLAŞTIRMA işlemi uygular

        Parameters  :   df = Veri setini içerir
                        column_name =  Standartlaştırma uygulanacak alandır
                        min =  Standarlaştırma aralığında yer alan alt değer
                        max =  Standarlaştırma aralığında yer alan üst değer
                        time = Standartlaştırılacak alanın Gün/ay/Yıl gibi zaman içerip içermediği bilgisi

        Returns     :   Fonskiyon geriye "_scaled" ismiyle standartlaştırılmış bir alan ekleyerek df'i döndürür.

        Example     :   standardization(df, "day_diff", 1, 5, "Yes") : 1-5 arasına günleri standarlaştırır.
                            "Yes" : 5 en yakın zaman demektir
                            "No"  : 5 en yüksek değer demektir
    """
    from sklearn.preprocessing import MinMaxScaler

    if time == "Yes":
        df[column_name + "_scaled"] = 5 - MinMaxScaler(feature_range=(min, max)). \
            fit(df[[column_name]]). \
            transform(df[[column_name]])
    else:
        df[column_name + "_scaled"] = MinMaxScaler(feature_range=(min, max)). \
            fit(df[[column_name]]). \
            transform(df[[column_name]])

    return df

# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


