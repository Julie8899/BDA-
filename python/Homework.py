import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1 Load the above data into a Pandas DataFrame
heads = (
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm",
    "class",
)
Iris_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=heads,
)
Iris_df1 = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

nd_array_1 = np.array(Iris_df1)[:, 0:4]

# 2-1 Mean
print(nd_array_1.mean(axis=0))

# 2-2 Min and Max
print(nd_array_1.min(axis=0))
print(nd_array_1.max(axis=0))

# 2-3 Quartiles
Lower = np.quantile(nd_array_1, 0.25, axis=0, interpolation="lower")
Higher = np.quantile(nd_array_1, 0.75, axis=0, interpolation="higher")
print(Lower)
print(Higher)

# 2-3-1 The other way for Quartile
Percent = np.percentile(nd_array_1, [25, 50, 75], axis=0)
print(Percent)

# 3-1 Scatter Plot

fig = px.scatter(
    Iris_df,
    x="sepal length in cm",
    y="sepal width in cm",
    color="class",
    size="petal width in cm",
    hover_data=["petal width in cm"],
    symbol="class",
)
fig.show()

# From the scatter plot, Senoto has higher sepal width but lower sepal length compare to other two kinds
# Scatter plot is easy to tell the difference if the difference is obvious among the kinds.

# 3-2 Violin Plot

df = px.data.tips()
fig = px.violin(
    Iris_df,
    y="petal width in cm",
    color="class",
    violinmode="overlay",  # draw violins on top of each other
    # default violinmode is 'group' as in example above
    hover_data=Iris_df.columns,
)
fig.show()

# Virginica has longest pedal width, but Senoto is lowest
# Violin plot is very straightforward but it only describe one predictor in one plot.

# 3-3 3D Scatter Plot

fig = px.scatter_3d(
    Iris_df,
    x="sepal length in cm",
    y="sepal width in cm",
    z="petal length in cm",
    color="petal width in cm",
    symbol="class",
)

fig.show()

# Virginica has higher sepeal width/lenghth and petal width/length than other two kinds.
# 3D scatter Plot can discribe more characters than my first scatter plot

# 3-4 3D bubble

fig = px.scatter_3d(
    Iris_df,
    x="sepal length in cm",
    y="sepal width in cm",
    z="petal length in cm",
    size="petal width in cm",
    color="class",
    hover_data=["class"],
)

fig.show()
# It is similar with 3D Scatter plot.

# 3-5 Polar Charts

fig = px.scatter_polar(
    Iris_df, r="petal length in cm", theta="class", size="petal width in cm"
)
fig.show()

# Viginica's petal length and width are longer than other two.
# These three kinds are distributed in different radius size round.
# so it is easy to see how much difference about petal among three different kinds

# 3-6 PCA Visualization in Python

fig = px.scatter_matrix(Iris_df, dimensions=heads, color="class")
fig.update_traces(diagonal_visible=False)
fig.show()

# PCA Visulization can help us to check the distribution of three kinds under different characters.
# So based on the plot, we can know Virginica always has the longest length and widest width for Sepal and petal

# 4 Use the StandardScaler transformer
# 4-1 DataFrame to numpy values

X_orig = Iris_df[
    [
        "sepal length in cm",
        "sepal width in cm",
        "petal length in cm",
        "petal width in cm",
    ]
].values
y = Iris_df["class"].values

# 4-2 Generate a feature from the where they started

scaler = StandardScaler()
scaler.fit(X_orig)
X = scaler.transform(X_orig)

# 4-3 Fit the features to a random forest

random_forest = RandomForestClassifier(random_state=100)
random_forest.fit(X, y)

test_df = Iris_df.iloc[:, 0:4]

print(test_df)

X_test_orig = test_df.values
X_test = scaler.transform(X_test_orig)
prediction = random_forest.predict(X_test)
probability = random_forest.predict_proba(X_test)

print(f"Classes: {random_forest.classes_}")
print(f"Probability: {probability}")
print(f"Predictions: {prediction}")

# 4-4 Fit the features to a Decision Tree

Decision_tree = tree.DecisionTreeClassifier(random_state=12)
Decision_tree.fit(X, y)

test_df = Iris_df.iloc[:, 0:4]

print(test_df)

X_test_orig = test_df.values
X_test = scaler.transform(X_test_orig)
prediction = Decision_tree.predict(X_test)
probability = Decision_tree.predict_proba(X_test)

print(f"Classes: {Decision_tree.classes_}")
print(f"Probability: {probability}")
print(f"Predictions: {prediction}")

# 5-1 Wrap the steps into a pipeline for random forest

pipeline = Pipeline(
    [
        ("StandardScaler", StandardScaler()),
        ("RandomForest", RandomForestClassifier(random_state=100)),
    ]
)
pipeline.fit(X_orig, y)

probability = pipeline.predict_proba(X_test_orig)
prediction = pipeline.predict(X_test_orig)
print(f"Probability: {probability}")
print(f"Predictions: {prediction}")

# 5-2 Wrap the steps into a pipeline for Decision Tree

pipeline = Pipeline(
    [
        ("StandardScaler", StandardScaler()),
        ("DecisionTree", tree.DecisionTreeClassifier(random_state=12)),
    ]
)
pipeline.fit(X_orig, y)

probability = pipeline.predict_proba(X_test_orig)
prediction = pipeline.predict(X_test_orig)
print(f"Probability: {probability}")
print(f"Predictions: {prediction}")
