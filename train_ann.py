from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

RANDOM_STATE, TEST_SIZE = 42, 0.20
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT, MODELS = HERE / "outputs", HERE / "models"
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
CANDIDATES = [ROOT/"Data_Preparation"/"telco_preprocessed.csv", ROOT/"Data_Preparation"/"training_set.csv",
              ROOT/"telco_preprocessed.csv", ROOT/"training_set.csv",
              Path.cwd()/"Data_Preparation"/"telco_preprocessed.csv", Path.cwd()/"Data_Preparation"/"training_set.csv"]

def load_data():
    for p in CANDIDATES:
        if p.exists():
            df = pd.read_csv(p); print(f"Loaded {p}; shape={df.shape}"); return df, p
    raise FileNotFoundError("No telco_preprocessed.csv or training_set.csv found. Place a prepared CSV in Data_Preparation or edit CANDIDATES.")

def encode_target(y):
    if y.dtype == object:
        s = y.astype(str).str.strip().str.lower()
        m = {"yes":1,"no":0,"true":1,"false":0,"1":1,"0":0,"churn":1,"stayed":0,"not churn":0}
        z = s.map(m)
        if z.isna().any(): raise ValueError(f"Unknown Churn labels: {sorted(s.unique())}")
        return z.astype(int)
    return pd.to_numeric(y, errors="raise").astype(int)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df, source = load_data()
    df.columns = [str(c).strip() for c in df.columns]
    targets = [c for c in df.columns if c.lower() in ("churn","churn_label","exited")]
    if not targets: raise ValueError(f"Cannot find Churn target. Columns are: {list(df.columns)}")
    target = targets[0]
    y = encode_target(df[target])
    ids = [c for c in df.columns if c.lower() in ("customerid","customer_id","id")]
    X = df.drop(columns=[target] + ids).copy()
    for c in X.select_dtypes(include="object").columns:
        numeric = pd.to_numeric(X[c].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        n = X[c].notna().sum()
        if n and numeric.notna().sum()/n >= .95: X[c] = numeric
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    num = X_train.select_dtypes(include=["number","bool"]).columns.tolist()
    cat = [c for c in X_train.columns if c not in num]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num),
        ("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam", alpha=.001,
                        learning_rate_init=.001, max_iter=300, early_stopping=True,
                        validation_fraction=.15, n_iter_no_change=20, random_state=RANDOM_STATE)
    model = Pipeline([("preprocessing",prep),("ann",ann)])
    print("Training ANN..."); model.fit(X_train,y_train)
    pred = model.predict(X_test); prob = model.predict_proba(X_test)[:,1]
    metrics = {"dataset":str(source),"target":target,"rows":int(len(df)),"train_rows":len(X_train),"test_rows":len(X_test),
      "predictors":X.columns.tolist(),"architecture":{"hidden_layers":[64,32],"activation":"ReLU","output":"Churn probability; threshold 0.50","optimizer":"Adam","loss":"binary cross-entropy/log loss","early_stopping":True},
      "test_metrics":{"accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision_score(y_test,pred,zero_division=0)),
      "recall":float(recall_score(y_test,pred,zero_division=0)),"f1_score":float(f1_score(y_test,pred,zero_division=0)),
      "roc_auc":float(roc_auc_score(y_test,prob)) if y_test.nunique()==2 else None},
      "classification_report":classification_report(y_test,pred,output_dict=True,zero_division=0),"iterations":int(model.named_steps["ann"].n_iter_)}
    joblib.dump(model,MODELS/"ann_churn_pipeline.joblib")
    (OUT/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    pd.DataFrame({"actual_churn":y_test.to_numpy(),"predicted_churn":pred,"predicted_churn_probability":prob}).to_csv(OUT/"test_predictions.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=["No Churn","Churn"],ax=ax); ax.set_title("ANN Test Confusion Matrix"); fig.tight_layout(); fig.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close(fig)
    if y_test.nunique()==2:
        fig,ax=plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(y_test,prob,ax=ax); ax.set_title("ANN Test ROC Curve"); fig.tight_layout(); fig.savefig(OUT/"roc_curve.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(range(1,len(ann.loss_curve_)+1),ann.loss_curve_); ax.set_xlabel("Iteration"); ax.set_ylabel("Loss"); ax.set_title("ANN Training Loss"); fig.tight_layout(); fig.savefig(OUT/"training_loss.png",dpi=160); plt.close(fig)
    print("\nTEST RESULTS")
    for k,v in metrics["test_metrics"].items(): print(f"{k}: {v:.4f}" if isinstance(v,(int,float)) and v is not None else f"{k}: {v}")
    print("Model and evaluation files saved in Predictive_Modeling/models and outputs.")

if __name__ == "__main__": main()
