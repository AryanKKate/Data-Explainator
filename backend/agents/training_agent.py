from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from imblearn.over_sampling import SMOTE

import numpy as np


class TrainingAgent:

    @staticmethod
    def train(
        df,
        target,
        model_plan
    ):

        # =========================
        # Split features and target
        # =========================

        y = df[target]

        X = df.drop(
            columns=[target]
        )

        assert target not in X.columns

        task = model_plan["task"]

        # ==================================
        # Detect binary vs multiclass
        # ==================================

        num_classes = y.nunique()

        is_binary = (
            num_classes == 2
        )

        # ==================================
        # Stratified split for classification
        # ==================================

        stratify = (
            y if task == "classification"
            else None
        )

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify

            )

        )

        # ==================================
        # Apply SMOTE ONLY on training data
        # ==================================

        if (

            task == "classification"

            and

            "smote" in model_plan.get(
                "steps",
                []
            )

            and

            y_train.nunique() > 1

        ):

            try:

                min_class_count = (
                    y_train.value_counts().min()
                )

                if min_class_count >= 2:

                    smote = SMOTE(
                        random_state=42,
                        k_neighbors=min(
                            5,
                            min_class_count - 1
                        )
                    )

                    X_train, y_train = (
                        smote.fit_resample(
                            X_train,
                            y_train
                        )
                    )

                    print(
                        "\nSMOTE applied ONLY on training data"
                    )

            except Exception as e:

                print(
                    f"\nSMOTE skipped: {e}"
                )

        # ==================================
        # Debug distributions
        # ==================================

        print("\nTrain class distribution:")
        print(y_train.value_counts())

        print("\nTest class distribution:")
        print(y_test.value_counts())

        # ==================================
        # Model storage
        # ==================================

        results = {}

        best_model = None

        best_score = -999

        available_models = {}

        # ==================================
        # Classification models
        # ==================================

        if task == "classification":

            available_models = {

                "random_forest":
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=10
                ),

                "logistic_regression":
                LogisticRegression(
                    random_state=42,
                    max_iter=3000
                )

            }

            # XGBoost
            try:

                from xgboost import XGBClassifier

                if is_binary:

                    available_models[
                        "xgboost"
                    ] = XGBClassifier(
                        random_state=42,
                        eval_metric="logloss"
                    )

                else:

                    available_models[
                        "xgboost"
                    ] = XGBClassifier(
                        random_state=42,
                        objective="multi:softprob",
                        num_class=num_classes,
                        eval_metric="mlogloss"
                    )

            except:
                pass

            # LightGBM
            try:

                from lightgbm import LGBMClassifier

                available_models[
                    "lightgbm"
                ] = LGBMClassifier(
                    random_state=42,
                    min_data_in_leaf=1,
                    min_data_in_bin=1,
                    verbosity=-1
                )

            except:
                pass

            # CatBoost
            try:

                from catboost import CatBoostClassifier

                available_models[
                    "catboost"
                ] = CatBoostClassifier(
                    verbose=0,
                    random_state=42
                )

            except:
                pass

        # ==================================
        # Regression models
        # ==================================

        elif task == "regression":

            available_models = {

                "random_forest":
                RandomForestRegressor(
                    random_state=42,
                    n_estimators=200,
                    max_depth=10
                ),

                "linear_regression":
                LinearRegression()

            }

        # ==================================
        # Training loop
        # ==================================

        for model_name in model_plan["models"]:

            if model_name not in available_models:

                continue

            model = available_models[
                model_name
            ]

            print(
                f"\nTraining {model_name}"
            )

            try:

                # ======================
                # Train
                # ======================

                model.fit(
                    X_train,
                    y_train
                )

                # ======================
                # Predict
                # ======================

                pred = model.predict(
                    X_test
                )

                # ====================================================
                # CLASSIFICATION
                # ====================================================

                if task == "classification":

                    # ----------------------------------
                    # Binary vs multiclass averaging
                    # ----------------------------------

                    average_type = (
                        "binary"
                        if is_binary
                        else "weighted"
                    )

                    accuracy = accuracy_score(
                        y_test,
                        pred
                    )

                    precision = precision_score(
                        y_test,
                        pred,
                        average=average_type,
                        zero_division=0
                    )

                    recall = recall_score(
                        y_test,
                        pred,
                        average=average_type,
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test,
                        pred,
                        average=average_type,
                        zero_division=0
                    )

                    result = {

                        "accuracy":
                        round(
                            accuracy,
                            3
                        ),

                        "precision":
                        round(
                            precision,
                            3
                        ),

                        "recall":
                        round(
                            recall,
                            3
                        ),

                        "f1":
                        round(
                            f1,
                            3
                        )

                    }

                    # ==================================
                    # ROC AUC
                    # ==================================

                    try:

                        if hasattr(
                            model,
                            "predict_proba"
                        ):

                            probs = (
                                model.predict_proba(
                                    X_test
                                )
                            )

                            # Binary
                            if is_binary:

                                roc_auc = (
                                    roc_auc_score(
                                        y_test,
                                        probs[:, 1]
                                    )
                                )

                            # Multiclass
                            else:

                                roc_auc = (
                                    roc_auc_score(
                                        y_test,
                                        probs,
                                        multi_class="ovr",
                                        average="weighted"
                                    )
                                )

                            result[
                                "roc_auc"
                            ] = round(
                                roc_auc,
                                3
                            )

                    except Exception as e:

                        print(
                            f"ROC AUC failed: {e}"
                        )

                    # ==================================
                    # Cross validation
                    # ==================================

                    if model_plan.get(
                        "cross_validation",
                        False
                    ):

                        try:

                            cv = StratifiedKFold(
                                n_splits=3,
                                shuffle=True,
                                random_state=42
                            )

                            scoring_metric = (

                                "f1"

                                if is_binary

                                else

                                "f1_weighted"

                            )

                            cv_scores = (

                                cross_val_score(

                                    model,

                                    X,

                                    y,

                                    cv=cv,

                                    scoring=scoring_metric

                                )

                            )

                            result[
                                "cv_f1_mean"
                            ] = round(
                                cv_scores.mean(),
                                3
                            )

                            result[
                                "cv_f1_std"
                            ] = round(
                                cv_scores.std(),
                                3
                            )

                        except Exception as e:

                            print(
                                f"CV failed: {e}"
                            )

                    # ==================================
                    # Save results
                    # ==================================

                    results[
                        model_name
                    ] = result

                    # ==================================
                    # Model selection logic
                    # ==================================

                    selection_score = (

                        result.get(
                            "cv_f1_mean",
                            f1
                        )

                    )

                    if selection_score > best_score:

                        best_score = (
                            selection_score
                        )

                        best_model = (
                            model_name
                        )

                # ====================================================
                # REGRESSION
                # ====================================================

                elif task == "regression":

                    rmse = np.sqrt(

                        mean_squared_error(
                            y_test,
                            pred
                        )

                    )

                    mae = mean_absolute_error(
                        y_test,
                        pred
                    )

                    r2 = r2_score(
                        y_test,
                        pred
                    )

                    result = {

                        "rmse":
                        round(
                            rmse,
                            3
                        ),

                        "mae":
                        round(
                            mae,
                            3
                        ),

                        "r2":
                        round(
                            r2,
                            3
                        )

                    }

                    # ==================================
                    # Cross validation
                    # ==================================

                    if model_plan.get(
                        "cross_validation",
                        False
                    ):

                        try:

                            cv = KFold(
                                n_splits=3,
                                shuffle=True,
                                random_state=42
                            )

                            cv_scores = (
                                cross_val_score(
                                    model,
                                    X,
                                    y,
                                    cv=cv,
                                    scoring="r2"
                                )
                            )

                            result[
                                "cv_r2_mean"
                            ] = round(
                                cv_scores.mean(),
                                3
                            )

                            result[
                                "cv_r2_std"
                            ] = round(
                                cv_scores.std(),
                                3
                            )

                        except Exception as e:

                            print(
                                f"CV failed: {e}"
                            )

                    results[
                        model_name
                    ] = result

                    selection_score = (

                        result.get(
                            "cv_r2_mean",
                            r2
                        )

                    )

                    if selection_score > best_score:

                        best_score = (
                            selection_score
                        )

                        best_model = (
                            model_name
                        )

            except Exception as e:

                print(
                    f"\n{model_name} failed: {e}"
                )

        # ==================================
        # Final Output
        # ==================================

        return {

            "best_model":
            best_model,

            "results":
            results

        }