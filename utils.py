import pandas as pd
import ast


# df = pd.DataFrame({
#         "vectors":[0.1, 0.2, 0.3],
#         "text": ["test1", "test2", "test3"]
# })
#
# df.to_csv("embeddings.csv", index=True)

df = pd.read_csv("embeddings.csv")
embeddings = df["vectors"].apply(lambda x: list(map(float, ast.literal_eval(x)))).to_list()
print(embeddings)