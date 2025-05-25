import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import mlflow

def preprocessing_data(filepath, output_dir):
    print(f"Memulai preprocessing untuk file: {filepath}")

    # Load dataset
    try:
        df = pd.read_csv(filepath)
        print("Dataset berhasil dimuat.")
        print(f"Bentuk dataset mentah: {df.shape}")
    except Exception as e:
        print(f"Error saat membaca file: {e}")
        return None

    # Validasi kolom target
    if 'quality' not in df.columns:
        print("Kolom 'quality' tidak ditemukan dalam dataset.")
        return None

    # Pisahkan fitur dan target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Scaling fitur
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Gabungkan kembali fitur yang sudah di-scale dengan target
    df_processed = X_scaled.copy()
    df_processed['quality'] = y.values

    # Simpan hasil preprocessing ke file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_winequality-red.csv")
    df_processed.to_csv(output_file, index=False)

    print(f"Data hasil preprocessing berhasil disimpan di: {output_file}")
    print(df_processed.head())

    return {
        "rows_clean": df_processed.shape[0],
        "files": [output_file]
    }

if __name__ == "__main__":
    input_file = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "dataset_raw/winequality-red.csv")
    output_dir = os.path.join(os.getenv("GITHUB_WORKSPACE", "."), "preprocessing/output")

    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Siapkan direktori MLflow
    mlruns_path = os.path.join(output_dir, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    mlflow.set_experiment("Preprocessing_Experiment")

    with mlflow.start_run(run_name="Preprocessing_Run"):
        result = preprocessing_data(input_file, output_dir)

        if result:
            mlflow.log_param("input_file", input_file)
            mlflow.log_param("output_file", result["files"][0])
            mlflow.log_metric("rows_clean", result["rows_clean"])

            for f in result["files"]:
                mlflow.log_artifact(f)
                print(f"Artefak berhasil di-log ke MLflow: {f}")
        else:
            print("Preprocessing gagal. Tidak ada file untuk di-log.")
