import json
import pandas as pd
from joblib import load


def handler_predict(event, _):
    '''
    Function that makes prediction from user data
    Args:
        event: {
            'users': {
                        user_1 {dict of features of user 1},
                        user_n {dict of features of user n}
            },
            'model_path': string,
            'threshold': float
                    }
    '''
    data_to_predict = pd.DataFrame.from_dict(json.loads(event["users"])).T

    model = load(event['model_path'])
    probs = model.predict_proba(data_to_predict)

    predictions = (probs[:, 1] >= event['threshold']).astype(int)

    user_names = data_to_predict.index.tolist()

    results = {}
    for i, user in enumerate(user_names):
        results[user] = int(predictions[i])

    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": results})
    }
