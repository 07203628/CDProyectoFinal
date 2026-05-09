import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


DEFAULT_TARGET = 'log_price'
DEFAULT_NUMERIC_FEATURES = [
    'accommodates',
    'bathrooms',
    'cleaning_fee',
    'host_response_rate_pct',
    'latitude',
    'longitude',
    'number_of_reviews',
    'review_scores_rating',
    'bedrooms',
    'beds',
    'host_since_year',
    'first_review_year',
    'last_review_year',
]
DEFAULT_CATEGORICAL_FEATURES = [
    'property_type',
    'room_type',
    'bed_type',
    'cancellation_policy',
    'city',
    'instant_bookable',
    'host_has_profile_pic',
    'host_identity_verified',
]

warnings.filterwarnings('ignore', message='Found unknown categories*')


def _split_csv_list(value):
    if not value:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def compute_vif(X):
    if X is None or X.empty:
        return pd.DataFrame(columns=['feature', 'VIF'])

    numeric = X.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.DataFrame(columns=['feature', 'VIF'])

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(numeric), columns=numeric.columns)
    if numeric.shape[1] == 0:
        return pd.DataFrame(columns=['feature', 'VIF'])

    X_const = sm.add_constant(numeric)
    vif_rows = []
    for idx, feature in enumerate(numeric.columns):
        try:
            value = variance_inflation_factor(X_const.values, idx + 1)
        except Exception:
            value = np.nan
        vif_rows.append({'feature': feature, 'VIF': value})
    return pd.DataFrame(vif_rows)


def _build_pipeline(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError('No features available to train the model.')

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('lr', LinearRegression()),
    ])
    return model


def _feature_names_from_pipeline(model, numeric_features, categorical_features):
    feature_names = list(numeric_features)

    if categorical_features:
        encoder = model.named_steps['preprocessor'].named_transformers_.get('cat')
        if encoder is not None:
            onehot = encoder.named_steps['onehot']
            feature_names.extend(onehot.get_feature_names_out(categorical_features).tolist())

    return feature_names


def train_model(
    csv_path,
    target=DEFAULT_TARGET,
    numeric_features=None,
    categorical_features=None,
    test_size=0.2,
    random_state=42,
    model_out='model.joblib',
):
    df = pd.read_csv(csv_path, low_memory=False)

    if target not in df.columns:
        raise ValueError(f'Target column "{target}" not found in dataset.')

    numeric_features = numeric_features or DEFAULT_NUMERIC_FEATURES
    categorical_features = categorical_features or DEFAULT_CATEGORICAL_FEATURES

    available_numeric = [col for col in numeric_features if col in df.columns]
    available_categorical = [col for col in categorical_features if col in df.columns]

    selected_columns = available_numeric + available_categorical + [target]
    model_df = df[selected_columns].copy()
    model_df[target] = pd.to_numeric(model_df[target], errors='coerce')
    model_df = model_df.dropna(subset=[target])

    X = model_df[available_numeric + available_categorical]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    vif = compute_vif(X_train[available_numeric]) if available_numeric else pd.DataFrame(columns=['feature', 'VIF'])

    pipe = _build_pipeline(available_numeric, available_categorical)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = len(available_numeric) + len(available_categorical)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    price_true = np.exp(y_test)
    price_pred = np.exp(y_pred)
    price_mse = mean_squared_error(price_true, price_pred)
    price_rmse = np.sqrt(price_mse)
    price_mae = mean_absolute_error(price_true, price_pred)

    coefs = pipe.named_steps['lr'].coef_
    feature_names = _feature_names_from_pipeline(pipe, available_numeric, available_categorical)
    coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
    coef_df['abs'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs', ascending=False)

    joblib.dump(pipe, model_out)

    results = {
        'r2': r2,
        'adj_r2': adj_r2,
        'mse': mse,
        'rmse': rmse,
        'price_mse': price_mse,
        'price_rmse': price_rmse,
        'price_mae': price_mae,
        'vif': vif,
        'coef_df': coef_df,
        'numeric_features': available_numeric,
        'categorical_features': available_categorical,
        'y_test_log': y_test.reset_index(drop=True),
        'y_pred_log': pd.Series(y_pred),
        'price_true': pd.Series(price_true.values),
        'price_pred': pd.Series(price_pred),
    }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Airbnb price model')
    parser.add_argument('--input', required=True, help='Path to cleaned CSV')
    parser.add_argument('--target', default=DEFAULT_TARGET, help='Target column name (default: log_price)')
    parser.add_argument('--numeric-features', default='', help='Comma-separated numeric features')
    parser.add_argument('--categorical-features', default='', help='Comma-separated categorical features')
    parser.add_argument('--output-model', default='model.joblib', help='Path to save trained model')
    args = parser.parse_args()

    numeric_features = _split_csv_list(args.numeric_features)
    categorical_features = _split_csv_list(args.categorical_features)

    res = train_model(
        args.input,
        target=args.target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_out=args.output_model,
    )

    print('R² (log_price):', res['r2'])
    print('R² ajustado (log_price):', res['adj_r2'])
    print('MSE (log_price):', res['mse'])
    print('RMSE (log_price):', res['rmse'])
    print('RMSE (precio real):', res['price_rmse'])
    print('MAE (precio real):', res['price_mae'])
    print('\nVIF:\n', res['vif'])
    print('\nCoeficientes (top ordenados por valor absoluto):\n', res['coef_df'].head(20))

    out_df = pd.DataFrame(
        {
            'log_price_true': res['y_test_log'],
            'log_price_pred': res['y_pred_log'],
            'price_true': res['price_true'],
            'price_pred': res['price_pred'],
        }
    )
    out_df.to_csv('predictions.csv', index=False)
    print('Predicciones guardadas en predictions.csv')
