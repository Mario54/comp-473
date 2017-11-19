from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from classify import load_data

all_date, classes = load_data("data-with-scanline-height.json")
select = SelectKBest(chi2, k=6)

select.fit(all_date, classes)
print(select.get_support())
