from pandas_profiling import ProfileReport


def getprofilereport(df):
    profile = ProfileReport(df)
    profile.to_file('Adult_Report.Html')
