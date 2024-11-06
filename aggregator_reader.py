import pandas as pd

array = [82, 123, 312]
df = pd.read_csv("data4.csv")
for i in range(1, 10):
    count_filter = df[df[f'Called in functions with {i} calls'] >3].shape[0]
    print(f"Number of functions called more than 3 times in functions with exactly {i} call:", count_filter )
    
    
import pandas as pd

df = pd.read_csv("data4.csv")

sort_columns = [f'Called in functions with {i} calls' for i in range(1, 10)]

df_sorted = df.sort_values(by=sort_columns, ascending=False)

df_sorted.to_csv("sorted_datav4.csv", index=False)

df_sorted = df_sorted[["Function Name"] + [f"Called in functions with {i} calls" for i in range(1, 6)]]

df_sorted.to_csv("sorted_data_v4.csv")