import pickle
import pandas as pd

filename = "co_T0_T1_toshare.pkl"
area_to_minority = "ks201ew_2011_oa_area_to_minority/KS201ew_2011_oa/KS201EWDATA.CSV"

with open(filename, "rb") as fle:
    df = pickle.load(fle)

new_df = df[['CompanyNumber', 'oa11']].copy()
new_df.columns = ['CompanyNumber', 'GeographyCode']
census_df = pd.read_csv(area_to_minority)

df2 = pd.merge(new_df, census_df, on='GeographyCode')
df2['MinorityScore'] = 1 - (df2['KS201EW0002'] / df2['KS201EW0001'])  # Proportion of ethnics in area
df3 = df2[['CompanyNumber', 'GeographyCode', 'MinorityScore']].copy()

print(df.shape[0]) # print length, for some reason there are 200,000 less entries in the created file
print(census_df.shape[0]) # probably due to some companies missing OA data
print(df3.shape[0])

df3.to_csv("OAtoMinorityData.csv")

