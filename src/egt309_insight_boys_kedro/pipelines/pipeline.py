from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_customers,
    clean_geolocation,
    clean_order_items,
    clean_order_reviews,
    clean_orders,
    clean_products,
    clean_sellers,
    clean_category_translation,
    clean_order_payments,
    merge_datasets,
    feature_engineering,
    split_train_val_test,
    balance_train_set,
    train_base_models,
    train_meta_learner,
    evaluate_stacked_model,
    save_stacking_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # === Data cleaning ===
            node(
                clean_customers,
                "raw_customers",
                "cleaned_customers",
                name="clean_customers",
            ),
            node(
                clean_geolocation,
                "raw_geolocation",
                "cleaned_geolocation",
                name="clean_geolocation",
            ),
            node(
                clean_order_items,
                "raw_order_items",
                "cleaned_order_items",
                name="clean_order_items",
            ),
            node(
                clean_order_reviews,
                "raw_order_reviews",
                "cleaned_order_reviews",
                name="clean_order_reviews",
            ),
            node(clean_orders, "raw_orders", "cleaned_orders", name="clean_orders"),
            node(
                clean_products,
                "raw_products",
                "cleaned_products",
                name="clean_products",
            ),
            node(clean_sellers, "raw_sellers", "cleaned_sellers", name="clean_sellers"),
            node(
                clean_category_translation,
                "raw_category_translation",
                "cleaned_category_translation",
                name="clean_category_translation",
            ),
            node(
                clean_order_payments,
                "raw_order_payments",
                "cleaned_order_payments",
                name="clean_order_payments",
            ),
            # === Merging ===
            node(
                merge_datasets,
                [
                    "cleaned_orders",
                    "cleaned_customers",
                    "cleaned_geolocation",
                    "cleaned_order_items",
                    "cleaned_order_payments",
                    "cleaned_order_reviews",
                    "cleaned_products",
                    "cleaned_sellers",
                    "cleaned_category_translation",
                ],
                "master_dataset",
                name="merge_datasets",
            ),
            # === Feature Engineering ===
            node(
                feature_engineering,
                "master_dataset",
                "feature_engineered_dataset",
                name="feature_engineering",
            ),
            # === Train/Validation/Test Split ===
            node(
                split_train_val_test,
                [
                    "feature_engineered_dataset",
                    "params:features",
                    "params:target_col",
                    "params:test_size",
                    "params:val_size",
                    "params:random_state",
                ],
                ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="split_train_val_test",
            ),
            # === Balance Train Set ===
            node(
                balance_train_set,
                ["X_train", "y_train"],
                ["X_bal", "y_bal"],
                name="balance_train_set",
            ),
            # === Train Base Models ===
            node(
                train_base_models,
                ["X_bal", "y_bal"],
                "base_models_output",
                name="train_base_models",
            ),
            # === Train Meta Learner ===
            node(
                train_meta_learner,
                ["base_models_output"],
                "meta_xgb",
                name="train_meta_learner",
            ),
            # === Evaluate Stacked Model ===
            node(
                evaluate_stacked_model,
                [
                    "meta_xgb",
                    "base_models_output",  # Now passing the entire dictionary
                    "X_val",
                    "y_val",
                    "X_test",
                    "y_test",
                ],
                "stacked_model_report",
                name="evaluate_stacked_model",
            ),
            # === Save Stacked Model ===
            node(
                save_stacking_model,
                ["stacked_model_report", "params:stacked_model_path"],
                "stacked_model_saved_path",
                name="save_stacked_model",
            ),
        ]
    )
