import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BOOLEAN_COLUMNS = [
    'cleaning_fee',
    'host_has_profile_pic',
    'host_identity_verified',
    'instant_bookable',
]

DATE_COLUMNS = ['host_since', 'first_review', 'last_review']
TEXT_DROP_COLUMNS = ['id', 'description', 'name', 'thumbnail_url', 'amenities', 'zipcode']
DERIVED_YEAR_COLUMNS = {
    'host_since': 'host_since_year',
    'first_review': 'first_review_year',
    'last_review': 'last_review_year',
}


def load_data(path):
    return pd.read_csv(path)


def _normalize_boolean(series):
    truthy = {'t', 'true', '1', 'yes', 'y'}
    falsy = {'f', 'false', '0', 'no', 'n'}

    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, bool):
            return int(value)
        normalized = str(value).strip().lower()
        if normalized in truthy:
            return 1
        if normalized in falsy:
            return 0
        return np.nan

    return series.map(convert).astype('float')


def _parse_percentage(series):
    return pd.to_numeric(
        series.astype(str).str.replace('%', '', regex=False).str.strip(),
        errors='coerce',
    )


def _add_derived_features(df):
    if 'log_price' in df.columns:
        df['price'] = np.exp(pd.to_numeric(df['log_price'], errors='coerce'))

    if 'host_response_rate' in df.columns:
        df['host_response_rate_pct'] = _parse_percentage(df['host_response_rate'])

    for source_col, target_col in DERIVED_YEAR_COLUMNS.items():
        if source_col in df.columns:
            parsed = pd.to_datetime(df[source_col], errors='coerce')
            df[target_col] = parsed.dt.year

    return df


def clean_data(df, drop_thresh=0.65, iqr_multiplier=1.5):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'log_price' not in df.columns:
        raise ValueError('The dataset must contain a log_price column.')

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = _normalize_boolean(df[col])

    if 'host_response_rate' in df.columns:
        df['host_response_rate_pct'] = _parse_percentage(df['host_response_rate'])

    df = _add_derived_features(df)

    df.drop(columns=TEXT_DROP_COLUMNS + DATE_COLUMNS, inplace=True, errors='ignore')

    df = df[df['log_price'].notna()].copy()

    q1 = df['log_price'].quantile(0.25)
    q3 = df['log_price'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    df = df[(df['log_price'] >= lower) & (df['log_price'] <= upper)]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else '')

    preferred = ['log_price', 'price']
    remaining = [col for col in df.columns if col not in preferred]
    ordered = [col for col in preferred if col in df.columns] + remaining
    return df[ordered]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean Airbnb Kaggle dataset')
    parser.add_argument('--input', required=False, help='Path to raw CSV')
    parser.add_argument('--output', required=False, help='Path to save cleaned CSV')
    args = parser.parse_args()

    default_input = 'Semana7/Proyecto/Final/Datos/raw.csv'
    default_output = 'Semana7/Proyecto/Final/Datos/cleaned.csv'

    input_path = args.input or default_input
    output_path = args.output or default_output

    if args.input is None and not Path(input_path).exists():
        print(f'Usando ruta por defecto: {input_path}')
    if args.output is None:
        print(f'Usando ruta por defecto de salida: {output_path}')

    try:
        df = load_data(input_path)
    except Exception as e:
        print(f'Error loading input file "{input_path}": {e}')
        raise

    cleaned = clean_data(df)
    cleaned.to_csv(output_path, index=False)
    print(f'Cleaned data saved to {output_path}')
