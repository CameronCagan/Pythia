import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging
   
def read_parquet_files(folder_path):
    dfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".parquet"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_parquet(full_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"ERROR reading {full_path}: {e}")
    if not dfs:
        raise ValueError("No parquet files found in folder or subfolders: " + folder_path)
    return pd.concat(dfs, ignore_index=True)

def read_csv_files(folder_path):
    dfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"ERROR reading {full_path}: {e}")
                    logging.error(f"ERROR reading CSV {full_path}: {e}")
    if not dfs:
        raise ValueError(f"No CSV files found in folder or subfolders: {folder_path}")
    return pd.concat(dfs, ignore_index=True)

def read_patient_outputs(folder_path):
    dfs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv") and not file.startswith("._"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)
                    dfs.append(df)
                except Exception as e:
                    logging.error(f"Error reading {full_path}: {e}")
    if not dfs:
        raise ValueError(f"No CSV output files found in {folder_path}")
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    df['yn'] = df['response'].str.lstrip('*').str.extract(r'(^.*?)[,.]')
    logging.info(f"YN Value Counts:\n{df.yn.value_counts()}\n")
    df['final_answer'] = df.groupby('empi')['yn'].transform(one_yes_nous)
    return df

def get_patient_data(df):
    df_pt = df[['empi', 'Ground Truth', 'final_answer']].drop_duplicates()
    logging.info(f"Patient-level Data Shape: {df_pt.shape}")
    logging.info(f"Answer Value Counts:\n{df_pt.final_answer.value_counts()}\n")
    return df_pt

def one_yes_nous(values):
    lower_values = [str(value).lower() for value in values]
    if 'yes' in lower_values:
        return 1
    else:
        return 0
    
def calculate_metrics(df, y_true_col="Ground Truth", y_pred_col="final_answer", output_file=None):
    y_true = df[y_true_col].astype(int)
    y_pred = df[y_pred_col].astype(int)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()

    n_pos = tp + fn
    n_neg = tn + fp

    sensitivity = tp / n_pos if n_pos > 0 else 1.0 
    specificity = tn / n_neg if n_neg > 0 else 1.0  
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    results = pd.DataFrame([{
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "F1 Score": round(f1, 4),
        "PPV": round(ppv, 4),
        "NPV": round(npv, 4),
    }])

    if output_file:
        results.to_csv(output_file, index=False)
        logging.info(f"Metrics saved to {output_file}")

    return float(sensitivity), float(specificity)

def full_save_fn_fp(
    df_pt,       
    visit_df,      
    fn_file, fn_list,
    fp_file, fp_list,
):

    fn_patients = df_pt.loc[
        (df_pt["Ground Truth"] == 1) &
        (df_pt["final_answer"] == 0),
        "empi"
    ].unique()

    logging.info(f"FN count: {len(fn_patients)}")

    fn_df = None

    if len(fn_patients) > 0:
        fn_rows = visit_df[visit_df["empi"].isin(fn_patients)].copy()

        logging.info(f"# FN Notes: {fn_rows.shape[0]}, Patients: {fn_rows['empi'].nunique()}")

        sort_cols = ["empi"]
        if "report_date_time" in fn_rows.columns:
            sort_cols.append("report_date_time")

        fn_rows = (
            fn_rows.sort_values(by=sort_cols)
                   .groupby("empi", group_keys=False)
                   .apply(lambda x: pd.concat([x.head(2), x.tail(2)]))
        )

        fn_rows.to_csv(fn_file, index=False)
        logging.info(f"FN combined notes saved to: {fn_file}")

        with open(fn_list, "w") as f:
            for pid in fn_patients:
                f.write(f"{pid}\n")

        logging.info(f"FN patient list saved to: {fn_list}")

        fn_df = fn_rows

    fp_patients = df_pt.loc[
        (df_pt["Ground Truth"] == 0) &
        (df_pt["final_answer"] == 1),
        "empi"
    ].unique()

    logging.info(f"FP count: {len(fp_patients)}")

    fp_df = None

    if len(fp_patients) > 0:
        fp_rows = visit_df[visit_df["empi"].isin(fp_patients)].copy()

        logging.info(f"# FP Notes: {fp_rows.shape[0]}, Patients: {fp_rows['empi'].nunique()}")

        sort_cols = ["empi"]
        if "report_date_time" in fp_rows.columns:
            sort_cols.append("report_date_time")

        fp_rows = fp_rows.sort_values(by=sort_cols)
        fp_rows.to_csv(fp_file, index=False)
        logging.info(f"FP notes saved to: {fp_file}")

        with open(fp_list, "w") as f:
            for pid in fp_patients:
                f.write(f"{pid}\n")

        logging.info(f"FP patient list saved to: {fp_list}")

        fp_df = fp_rows

    return fn_df, fp_df