import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


print(nd_array_1.mean(axis=0))
print(nd_array_1.min(axis=0))
print(nd_array_1.max(axis=0))
Lower = np.quantile(nd_array_1, 0.25, axis=0, interpolation="lower")
Higher = np.quantile(nd_array_1, 0.75, axis=0, interpolation="higher")
print(Lower)
print(Higher)

Percent = np.percentile(nd_array_1, [25, 50, 75], axis=0)  # The other way for Quartile
print(Percent)

# Scatter Plot

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


# Violin Plot

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

# 3D Scatter Plot

fig = px.scatter_3d(
    Iris_df,
    x="sepal length in cm",
    y="sepal width in cm",
    z="petal length in cm",
    color="petal width in cm",
    symbol="class",
)


fig.show()

# 3D bubble

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

#
fig = px.scatter_polar(
    Iris_df, r="petal length in cm", theta="class", size="petal width in cm"
)
fig.show()

# PCA Visualization in Python
fig = px.scatter_matrix(Iris_df, dimensions=heads, color="class")
fig.update_traces(diagonal_visible=False)
fig.show()

# DataFrame to numpy values
X_orig = Iris_df[
    [
        "sepal length in cm",
        "sepal width in cm",
        "petal length in cm",
        "petal width in cm",
    ]
].values
y = Iris_df["class"].values

# Let's generate a feature from the where they started
scaler = StandardScaler()
scaler.fit(X_orig)
X = scaler.transform(X_orig)

# Fit the features to a random forest


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

#
Decision_tree = tree.DecisionTreeClassifier(random_state=12)
Decision_tree.fit(X, y)

test_df = Iris_df.iloc[:, 0:4]

print(test_df)

X_test_orig = test_df.values
X_test = scaler.transform(X_test_orig)
prediction = Decision_tree.predict(X_test)
probability = Decision_tree.predict_proba(X_test)

print(f"Classes: {random_forest.classes_}")
print(f"Probability: {probability}")
print(f"Predictions: {prediction}")

# As Pipeline


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
