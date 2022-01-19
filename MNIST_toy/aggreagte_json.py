import json
import pandas as pd

import sys

file_names = sys.argv[1:] #get names from console

#print(file_names)

files = map(lambda x : open(x, "r"), file_names)
json_records = list(map(lambda x : json.loads(x.read()),  files ))

df = pd.DataFrame.from_records(json_records)

df.to_csv("test.csv")