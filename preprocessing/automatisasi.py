import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys
import mlflow

def preprocess_data(raw_data_path: str, processed_data_file_path: str): # Nama parameter diubah agar lebih jelas
    print(f"Memulai proses preprocessing untuk file: {raw_data_path}")

    # 1. Load Data
    try:
        df = pd.read_csv(raw_data_path)
        print("Dataset mentah berhasil dimuat.")
        print(f"Bentuk dataset mentah: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {raw_data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        sys.exit(1)

    if 'quality' not in df.columns:
        print("Error: Kolom target 'quality' tidak ditemukan dalam dataset.")
        sys.exit(1)

    # 2. Pisahkan fitur dan target
    X = df.drop('quality', axis=1)
    y = df['quality']
    feature_names = X.columns.tolist()

    print(f"Fitur (X) dipisahkan: {X.shape}, Target (y): {y.shape}")

    # 3. Scaling fitur
    print("Melakukan scaling pada fitur...")
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=feature_names)

    # 4. Gabungkan kembali dengan target
    df_processed = X_scaled.copy()
    df_processed['quality'] = y.values

    # 5. Simpan ke output path
    try:
        # Pastikan direktori untuk file output ada
        os.makedirs(os.path.dirname(processed_data_file_path), exist_ok=True)
        df_processed.to_csv(processed_data_file_path, index=False)
        print(f"Data berhasil disimpan di {processed_data_file_path}")
        print(df_processed.head())

        # Mengembalikan informasi untuk MLflow
        return {
            "rows_clean": len(df_processed),
            "files": [processed_data_file_path]
        }
    except Exception as e:
        print(f"Error saat menyimpan data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    base_path = os.environ.get("GITHUB_WORKSPACE", ".")

    input_file = os.path.join(base_path, "dataset_raw/winequality-red.csv")
    # Output directory untuk menyimpan data yang sudah diproses dan artefak MLflow
    output_dir = os.path.join(base_path, "preprocessing/output")

    # Tentukan nama file output untuk data yang diproses
    processed_output_file = os.path.join(output_dir, "processed_winequality-red.csv")

    print(f"Output directory: {output_dir}")
    print(f"Input file: {input_file}")
    print(f"Processed output file: {processed_output_file}")

    # Membuat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # Path untuk menyimpan data MLflow runs
    mlruns_path = os.path.join(output_dir, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    mlflow.set_experiment("Preprocessing_Experiment")

    with mlflow.start_run(run_name="Preprocessing_Run") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        result = preprocess_data(input_file, processed_output_file)

        if result: 
            mlflow.log_param("input_file", input_file)
            mlflow.log_param("output_data_path", processed_output_file)
            mlflow.log_metric("rows_clean", result["rows_clean"])

            # Log file yang diproses sebagai artefak
            for f_path in result["files"]:
                mlflow.log_artifact(f_path, artifact_path="processed_data") 
            print(f"Artefak {result['files']} berhasil di-log ke MLflow.")
        else:
            print("Preprocessing gagal, tidak ada hasil untuk di-log ke MLflow.")

    print("Proses preprocessing selesai.")