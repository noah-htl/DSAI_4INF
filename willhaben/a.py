import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# 1. Load Data (assuming headers are added)
df = pd.read_csv("data.csv", names=["id", "price", "brand", "zip", "region", "title"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df.dropna(subset=["price"], inplace=True)
df = df[df['price'] <= 5000]

# 2. Simple Feature Engineering
df['is_vintage'] = df['title'].str.contains('vintage|alt|antik', regex=True, case=False).astype(int)
df['is_herren'] = df['title'].str.contains('herren', regex=True, case=False).astype(int)
df['is_damen'] = df['title'].str.contains('damen', regex=True, case=False).astype(int)
df['title_length'] = df['title'].str.len().astype(int)
df['is_gold'] = df['title'].str.contains('gold|750|14k', regex=True, case=False).astype(int)
df['is_chrono'] = df['title'].str.contains('chrono|automatic|automatisch', regex=True, case=False).astype(int)
df['is_defekt'] = df['title'].str.contains('defekt|kaputt|bastler|repar', regex=True, case=False).astype(int)
luxury_brands = 'rolex|omega|breitling|cartier|iwc|tudor|patek|audemars|longines|heuer'
df['is_luxury'] = df['title'].str.contains(luxury_brands, regex=True, case=False).astype(int)

"""import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of prices
sns.histplot(np.log1p(df['price']), kde=True)
plt.title('Price Distribution (Log)')
plt.show()

# Check numerical skewness value (0 = perfectly normal, >1 = highly skewed)
print("Skewness:", df['price'].skew())
exit(0)"""

df['price'] = np.log1p(df['price'])

# 3. Handle Categorical Features
X = df[["brand","region",'is_vintage', 'is_damen', 'is_herren', 'title_length','is_gold', 'is_chrono', 'is_defekt', 'is_luxury']].copy()
y = df['price']
X['brand'] = X['brand'].astype('category')
X['region'] = X['region'].astype('category')

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = HistGradientBoostingRegressor(
    max_iter=200,          # Allow more trees to fix errors
    learning_rate=0.05,     # Slower learning rate for better generalization
    max_leaf_nodes=63,      # Deeper trees to capture complex interactions
    min_samples_leaf=15,    # Minimum samples per leaf
    random_state=42,
    categorical_features="from_dtype"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)


y_train_euros = np.expm1(y_train)
y_test_euros = np.expm1(y_test)

pred_train_euros = np.expm1(model.predict(X_train))
pred_test_euros = np.expm1(predictions)

# 6. Evaluate
print(f"MAE train: {mean_absolute_error(y_train_euros, pred_train_euros)}")
print(f"MAE test: {mean_absolute_error(y_test_euros, pred_test_euros)}")
print(f"RMSE: {root_mean_squared_error(y_test_euros, pred_test_euros)}")
print(f"R2: {r2_score(y_test_euros, pred_test_euros)}")