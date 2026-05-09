from Limpieza import load_data, clean_data

if __name__ == '__main__':
    input_path = 'Semana7/Proyecto/Final/Datos/raw.csv'
    output_path = 'Semana7/Proyecto/Final/Datos/cleaned.csv'
    df = load_data(input_path)
    cleaned = clean_data(df)
    cleaned.to_csv(output_path, index=False)
    print(f'Cleaned data saved to {output_path}')
