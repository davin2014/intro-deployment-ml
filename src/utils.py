

from matplotlib import pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
import seaborn as sns


def update_model(model: Pipeline) -> None:
    """
    This function update the model with new data
    
    Args:
        model: The model to be updated
    """
    dump(model, 'model/model.pkl')


def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    """
    This function save the metrics report
    
    Args:
        train_score: The train score
        test_score: The test score
        validation_score: The validation score
        model: The model to be updated
    """
    with open('report.txt', 'w') as outfile:

        outfile.write(f'# Model Pipeline Description Report\n') 

        for key, value in model.named_steps.items():
            outfile.write(f'### {key}: {value.__repr__()}\n')

        outfile.write(f'### train_score: {train_score}\n')
        outfile.write(f'### test_score: {test_score}\n')
        outfile.write(f'### validation_score: {validation_score}\n')
        
def get_model_performance_set(y_test: pd.Series, y_test_pred: pd.Series) -> None:
    """
    This function save the metrics report
    Args:
        y_test: The test target
        y_test_pred: The test prediction
    """
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_test_pred, y= y_test, ax=ax)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('Behavior model prediction')
    fig.savefig('predicrion_behavior.png')