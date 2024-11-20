import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from catboost import CatBoostRegressor, Pool
from bayes_opt import BayesianOptimization
from lofo import LOFOImportance, Dataset, plot_importance

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Load Datasets
train = pd.read_csv("./phone_train.csv")
test = pd.read_csv("./phone_validation.csv")

# Define Categorical and Numerical Columns
categorical_cols = ['tariff.plan', 'payment.method', 'activation.zone', 'activation.channel', 'vas1', 'vas2', 'sex']
numerical_cols = sorted(list(set(train.columns) - set(categorical_cols) - {'y'} - {'age'}))

# Preprocess Numerical Columns
train_log1p = train.copy()
train_log1p[numerical_cols] = train_log1p[numerical_cols].clip(lower=0)
train_log1p[numerical_cols] = np.log1p(train_log1p[numerical_cols])
scaler = StandardScaler()
train_log1p[numerical_cols] = scaler.fit_transform(train_log1p[numerical_cols])

test_log1p = test.copy()
test_log1p[numerical_cols] = test_log1p[numerical_cols].clip(lower=0)
test_log1p[numerical_cols] = np.log1p(test_log1p[numerical_cols])
test_log1p[numerical_cols] = scaler.transform(test_log1p[numerical_cols])

# Apply Log Transformation to the Target Variable
y_train = np.log1p(train['y'])

# Encode Categorical Features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_encoded = encoder.fit_transform(train[categorical_cols])
test_encoded = encoder.transform(test[categorical_cols])

# Convert Encoded Features to DataFrames
encoded_cols = encoder.get_feature_names_out(categorical_cols)
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test.index)

numerical_cols.append('age')
# Concatenate Numerical and Encoded Categorical Features
train_final = pd.concat([pd.DataFrame(train_log1p[numerical_cols], index=train.index), train_encoded_df], axis=1)
test_final = pd.concat([pd.DataFrame(test_log1p[numerical_cols], index=test.index), test_encoded_df], axis=1)

# Ensure Target Column is Included in the Sample Data
train_final['y'] = np.log1p(train['y'])  # Make sure the target is log-transformed

# Extract a Sample of the Data
sample_df = train_final.sample(frac=0.01, random_state=0)
sample_df.sort_values("age", inplace=True)  # Sort by a feature for time split validation

# Define the Validation Scheme
cv = KFold(n_splits=4, shuffle=False, random_state=None)  # Don't shuffle to keep the time split validation

# Define the Binary Target and the Features
dataset = Dataset(df=sample_df, target='y', features=[col for col in train_final.columns if col != 'y'])

# Define the Validation Scheme and Scorer. The default model is LightGBM
lofo_imp = LOFOImportance(dataset, cv=cv, scoring=rmse_scorer)

# Get the Mean and Standard Deviation of the Importances in Pandas Format
importance_df = lofo_imp.get_importance()

# Plot the Means and Standard Deviations of the Importances
plot_importance(importance_df, figsize=(12, 20))
important_features = importance_df[importance_df['importance_mean'] > 0]['feature'].tolist()

important_features = ['tariff.plan_4',
 'q09.out.dur.peak',
 'q05.out.ch.peak',
 'q07.ch.cc',
 'q05.out.val.offpeak',
 'q02.ch.cc',
 'q08.out.dur.peak',
 'q05.out.dur.offpeak',
 'q09.in.dur.tot',
 'q09.out.ch.peak',
 'activation.zone_1',
 'q07.in.ch.tot',
 'q04.out.dur.peak',
 'q07.out.dur.offpeak',
 'tariff.plan_8',
 'q09.out.val.offpeak',
 'q01.out.val.offpeak',
 'q06.ch.sms',
 'q07.in.dur.tot',
 'q08.ch.sms',
 'q03.ch.sms',
 'payment.method_Carta di Credito',
 'q02.out.ch.peak',
 'q07.out.dur.peak',
 'q03.out.val.peak',
 'activation.zone_3',
 'q08.out.val.peak',
 'sex_F',
 'vas2_Y',
 'q06.out.ch.offpeak',
 'q02.out.dur.peak',
 'activation.channel_3',
 'q01.out.dur.offpeak',
 'q03.out.dur.peak']

# Filter Training and Test Data to Keep Only Important Features
train_filtered = train_final[important_features]
test_filtered = test_final[important_features]

# Define Parameter Space for Bayesian Optimization
def catboost_evaluate(depth, learning_rate, l2_leaf_reg, iterations, border_count, bagging_temperature):
    params = {
        'depth': int(depth),
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'iterations': int(iterations),
        'border_count': int(border_count),
        'bagging_temperature': bagging_temperature,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'random_seed': 32,
        'logging_level': 'Silent',
        'task_type': 'GPU'
    }
    model = CatBoostRegressor(**params)
    # Use a subset of the data for faster optimization
    sample_indices = np.random.choice(train_filtered.index, size=1000, replace=False)
    X_sample = train_filtered.loc[sample_indices]
    y_sample = y_train.loc[sample_indices]
    cv_results = cross_val_score(model, X_sample, y_sample, cv=3, scoring=rmse_scorer).mean()
    return cv_results  # Negative because BayesianOptimization maximizes

# Bayesian Optimization with Parameter Bounds
catboost_bo = BayesianOptimization(
    f=catboost_evaluate,
    pbounds={
        'depth': (5, 15),
        'learning_rate': (0.1, 0.2),
        'l2_leaf_reg': (1, 10),
        'iterations': (500, 1500),
        'border_count': (120, 255),
        'bagging_temperature': (0.0, 1.0),
    },
    random_state=42,
    verbose=2
)

catboost_bo.maximize(init_points=5, n_iter=25)

# Get the Best Parameters and Train the Final Model
best_params = catboost_bo.max['params']
best_params['depth'] = int(best_params['depth'])
best_params['iterations'] = int(best_params['iterations'])
best_params['border_count'] = int(best_params['border_count'])

model_final = CatBoostRegressor(
    **best_params,
    eval_metric='RMSE',
    loss_function='RMSE',
    random_seed=32,
    logging_level='Silent',
    task_type='GPU'
)

# Split the Filtered Training Data for Final Evaluation
X_train, X_val, y_train_split, y_val_split = train_test_split(train_filtered, y_train, test_size=0.2, random_state=42)

# Create CatBoost Pools
train_pool = Pool(data=X_train, label=y_train_split)
val_pool = Pool(data=X_val, label=y_val_split)

model_final.fit(train_pool, eval_set=val_pool, metric_period=10)

# Predict on the Validation Data
y_val_pred = model_final.predict(val_pool)

# Calculate and Print the RMSE on Validation Data
rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
print(f'Validation Log_RMSE: {rmse}')

# Predict on the Test Data
y_pred = model_final.predict(test_filtered)

# Reverse the Log Transformation
y_pred = np.maximum(0, np.expm1(y_pred))

# Save the Predictions
np.savetxt('/content/drive/MyDrive/final_predictions_catboost.txt', y_pred, fmt='%f')

# Save the Best Parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('/content/drive/MyDrive/best_params_catboost.csv', index=False)