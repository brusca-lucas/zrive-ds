from sklearn.ensemble import GradientBoostingClassifier
import json
import os
import boto3
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from joblib import dump


def data_loading():
    local_path = "/data/feature_frame.csv"
    load_dotenv()
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    s3 = session.client("s3")

    bucket_name = "zrive-ds-data"
    path = "groceries/box_builder_dataset/feature_frame.csv"

    try:
        s3.download_file(bucket_name, path, local_path)
        print("File downloaded succesfully.")
    except Exception as e:
        print(f"File not found : {e}")
    dataset = pd.read_csv(local_path)
    return dataset


def preprocessing(dataset):
    orders = dataset[dataset['outcome'] == 1]
    order_sizes = orders.groupby('order_id').size()
    large_orders = order_sizes[order_sizes >= 5].index
    selected_orders = dataset[dataset['order_id'].isin(large_orders)]
    cols = ['user_order_seq',
            'ordered_before',
            'abandoned_before',
            'active_snoozed',
            'set_as_regular',
            'normalised_price',
            'discount_pct',
            'global_popularity',
            'avg_days_to_buy_variant_id',
            'std_days_to_buy_variant_id',
            'avg_days_to_buy_product_type',
            'std_days_to_buy_product_type',
            'outcome'
            ]
    return selected_orders[cols]


def handler_fit(event: dict, _):
    """
    Function that trains a GradientBoostingClassifier model.

    Expected event structure:
    {
        "model_parametrisation": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
            },
        },
        "model_folder_path": "zrive-ds/models"
    }
    """
    model_parametrisation = event["model_parametrisation"]
    model_path = event['model_folder_path']
    model = GradientBoostingClassifier(**model_parametrisation)
    data = preprocessing(data_loading())
    model.fit(data.drop('outcome'), data['outcome'])

    today = datetime.now().strftime("%Y-%m-%d")
    model_name = f"gradient_boosting_model_{today}.pkl"

    full_model_path = os.path.join(model_path, model_name)
    dump(model, full_model_path)

    return {
            "statusCode": "200",
            "body": json.dumps(
                {"model_path": [model_path]})
    }
