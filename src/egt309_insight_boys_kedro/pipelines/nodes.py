import pandas as pd


# 1. Clean customers
def clean_customers(customers_df: pd.DataFrame) -> pd.DataFrame:
    customers_df = customers_df.rename(
        columns={
            "customer_zip_code_prefix": "zip_code",
            "customer_city": "city",
            "customer_state": "state",
        }
    )
    return customers_df


# 2. Clean geolocation
def clean_geolocation(geolocation_df: pd.DataFrame) -> pd.DataFrame:
    geolocation_df = geolocation_df.drop(columns=["geolocation_lat", "geolocation_lng"])
    geolocation_df = geolocation_df.rename(
        columns={
            "geolocation_zip_code_prefix": "zip_code",
            "geolocation_city": "city",
            "geolocation_state": "state",
        }
    )
    geolocation_df = geolocation_df.drop_duplicates()
    return geolocation_df


# 3. Clean order_items
def clean_order_items(order_items_df: pd.DataFrame) -> pd.DataFrame:
    order_items_df = order_items_df.drop(
        columns=["freight_value", "shipping_limit_date"]
    )
    return order_items_df


# 4. Clean order_reviews
def clean_order_reviews(order_reviews_df: pd.DataFrame) -> pd.DataFrame:
    order_reviews_df = order_reviews_df.drop(
        columns=[
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_answer_timestamp",
        ]
    )
    order_reviews_df = order_reviews_df.dropna()
    order_reviews_df = order_reviews_df.reset_index(drop=True)
    return order_reviews_df


# 5. Clean orders
def clean_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    orders_df = orders_df.drop(
        columns=[
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_estimated_delivery_date",
        ]
    )
    orders_df = orders_df.dropna()
    orders_df = orders_df.reset_index(drop=True)
    return orders_df


# 6. Clean products
def clean_products(products_df: pd.DataFrame) -> pd.DataFrame:
    products_df = products_df.drop(
        columns=[
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    )
    return products_df


# 7. Clean sellers
def clean_sellers(sellers_df: pd.DataFrame) -> pd.DataFrame:
    sellers_df = sellers_df.rename(
        columns={
            "seller_zip_code_prefix": "zip_code",
            "seller_city": "city",
            "seller_state": "state",
        }
    )
    return sellers_df


# 8. Clean product_category_name_translation
def clean_category_translation(category_translation_df: pd.DataFrame) -> pd.DataFrame:
    return category_translation_df  # No cleaning needed


# 9. Clean order_payments (no cleaning needed)
def clean_order_payments(order_payments_df: pd.DataFrame) -> pd.DataFrame:
    return order_payments_df


# 10. Merge all cleaned datasets
def merge_datasets(
    orders_df,
    customers_df,
    geolocation_df,
    order_items_df,
    order_payments_df,
    order_reviews_df,
    products_df,
    sellers_df,
    category_translation_df,
) -> pd.DataFrame:
    # 3.1 - 3.8: Merging as per your logic above

    master = pd.merge(orders_df, customers_df, on="customer_id", how="left")
    master = pd.merge(
        master, geolocation_df, on=["zip_code", "city", "state"], how="left"
    )
    master = pd.merge(master, order_items_df, on="order_id", how="left")
    master = pd.merge(master, order_payments_df, on="order_id", how="left")
    master = pd.merge(master, order_reviews_df, on="order_id", how="left")
    master = pd.merge(master, products_df, on="product_id", how="left")
    master = pd.merge(master, sellers_df, on="seller_id", how="left")
    master = pd.merge(
        master, category_translation_df, on="product_category_name", how="left"
    )

    master = master.drop(
        columns=[
            "customer_id",
            "zip_code_y",
            "city_y",
            "state_y",
            "order_item_id",
            "payment_sequential",
            "review_id",
            "product_category_name",
        ],
        errors="ignore",
    )

    master = master.rename(
        columns={"zip_code_x": "zip_code", "city_x": "city", "state_x": "state"}
    )

    # Fill nulls as described
    master = master.drop_duplicates()
    master["payment_type"] = master["payment_type"].fillna("unknown")
    master["payment_installments"] = master["payment_installments"].fillna(0)
    master["payment_value"] = master["payment_value"].fillna(0)
    mode_value = master["review_score"].mode()[0]
    master["review_score"] = master["review_score"].fillna(mode_value)
    master["product_category_name_english"] = master[
        "product_category_name_english"
    ].fillna("NaN")
    return master


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Compute customer frequency
    customer_orders = (
        df.groupby("customer_unique_id")["order_id"].nunique().reset_index()
    )
    customer_orders.columns = ["customer_unique_id", "order_count"]

    # 2. Merge back and create the target column
    df = pd.merge(df, customer_orders, on="customer_unique_id", how="left")
    df["is_repeat_buyer"] = df["order_count"].apply(lambda x: 1 if x > 1 else 0)

    # 3. Convert to datetime
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    # 4. Calculate recency
    last_purchase = (
        df.groupby("customer_unique_id")["order_purchase_timestamp"].max().reset_index()
    )
    now = df["order_purchase_timestamp"].max()
    last_purchase["recency_days"] = (
        now - last_purchase["order_purchase_timestamp"]
    ).dt.days
    df = pd.merge(
        df,
        last_purchase[["customer_unique_id", "recency_days"]],
        on="customer_unique_id",
        how="left",
    )

    # 5. Extract month and day of week
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    return df


def split_train_val_test(df, features, target_col, test_size, val_size, random_state):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    for col in ["product_category_name_english", "payment_type", "city", "state"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[features]
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    print("Train:", y_train.value_counts())
    print("Val:", y_val.value_counts())
    print("Test:", y_test.value_counts())

    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_train_set(X_train, y_train):
    from imblearn.over_sampling import SMOTE
    import pandas as pd

    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    print("After partial SMOTE:", pd.Series(y_bal).value_counts())

    return X_bal, y_bal


def train_base_models(X_bal, y_bal):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.ensemble import BalancedBaggingClassifier

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(X_bal), 5))  # 5 base models
    xgb_models, cat_models, rf_models, lgbm_models, bagging_models = [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_bal, y_bal)):
        print(f"Fold {fold + 1}")
        X_tr, X_val_fold = X_bal.iloc[train_idx], X_bal.iloc[val_idx]
        y_tr, y_val_fold = y_bal.iloc[train_idx], y_bal.iloc[val_idx]

        xgb = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", scale_pos_weight=1
        )
        xgb.fit(X_tr, y_tr)
        oof_preds[val_idx, 0] = xgb.predict_proba(X_val_fold)[:, 1]
        xgb_models.append(xgb)

        cat = CatBoostClassifier(verbose=0, scale_pos_weight=1)
        cat.fit(X_tr, y_tr)
        oof_preds[val_idx, 1] = cat.predict_proba(X_val_fold)[:, 1]
        cat_models.append(cat)

        rf = RandomForestClassifier(
            class_weight="balanced", n_estimators=200, random_state=42
        )
        rf.fit(X_tr, y_tr)
        oof_preds[val_idx, 2] = rf.predict_proba(X_val_fold)[:, 1]
        rf_models.append(rf)

        lgbm = LGBMClassifier(
            class_weight="balanced", n_estimators=200, random_state=42, verbosity=-1
        )
        lgbm.fit(X_tr, y_tr)
        oof_preds[val_idx, 3] = lgbm.predict_proba(X_val_fold)[:, 1]
        lgbm_models.append(lgbm)

        bagging = BalancedBaggingClassifier(
            estimator=RandomForestClassifier(),
            n_estimators=10,
            sampling_strategy="auto",
            random_state=42,
            n_jobs=-1,
        )
        bagging.fit(X_tr, y_tr)
        oof_preds[val_idx, 4] = bagging.predict_proba(X_val_fold)[:, 1]
        bagging_models.append(bagging)

    return {
        "oof_preds": oof_preds,
        "xgb_models": xgb_models,
        "cat_models": cat_models,
        "rf_models": rf_models,
        "lgbm_models": lgbm_models,
        "bagging_models": bagging_models,
        "y_bal": y_bal,
    }


def train_meta_learner(base_models_output):
    from xgboost import XGBClassifier

    oof_preds = base_models_output["oof_preds"]
    y_bal = base_models_output["y_bal"]

    meta_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=1,
        random_state=42,
    )
    meta_xgb.fit(oof_preds, y_bal)

    print("Meta-learner trained successfully.")
    return meta_xgb


def ensemble_predict(X, model_lists):
    import numpy as np

    pred_arr = []
    for models in model_lists:
        preds = [model.predict_proba(X)[:, 1] for model in models]
        pred_arr.append(np.mean(preds, axis=0))
    return np.column_stack(pred_arr)


def evaluate_stacked_model(meta_xgb, base_models_output, X_val, y_val, X_test, y_test):
    import numpy as np
    from sklearn.metrics import classification_report, precision_recall_curve

    model_lists = [
        base_models_output["xgb_models"],
        base_models_output["cat_models"],
        base_models_output["rf_models"],
        base_models_output["lgbm_models"],
        base_models_output["bagging_models"],
    ]

    val_base_preds = ensemble_predict(X_val, model_lists)
    test_base_preds = ensemble_predict(X_test, model_lists)

    val_probs = meta_xgb.predict_proba(val_base_preds)[:, 1]
    test_probs = meta_xgb.predict_proba(test_base_preds)[:, 1]

    prec, rec, thresh = precision_recall_curve(y_val, val_probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1)
    best_thresh = thresh[best_idx]

    print(f"Best threshold on val set: {best_thresh:.2f}")
    print(
        f"Precision: {prec[best_idx]:.3f}, Recall: {rec[best_idx]:.3f}, F1: {f1[best_idx]:.3f}"
    )

    val_preds = (val_probs >= best_thresh).astype(int)
    test_preds = (test_probs >= best_thresh).astype(int)

    print("==== VALIDATION SET ====")
    print(classification_report(y_val, val_preds))
    print("==== TEST SET ====")
    print(classification_report(y_test, test_preds))

    return {
        "best_thresh": best_thresh,
        "val_report": classification_report(y_val, val_preds, output_dict=True),
        "test_report": classification_report(y_test, test_preds, output_dict=True),
    }


def save_stacking_model(stacked_model_report, output_path: str):
    import joblib

    joblib.dump(stacked_model_report, output_path)
    return output_path
