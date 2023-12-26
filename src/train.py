from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from io import StringIO
import sys
import logging

from utils import update_model, save_simple_metrics_report, get_model_performance_set


logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr
        )

logger = logging.getLogger(__name__)

logging.info('Loading Data..........')
data = pd.read_csv('dataset/full_data.csv')

logging.info('loading model..........')

model = Pipeline([('imputer', SimpleImputer(strategy='mean',missing_values=np.nan)), 
                  ('core_model', GradientBoostingRegressor())
                  ])

logging.info('Seraparating daraset into train and test..........')
x = data.drop(['worldwide_gross'], axis=1)
y = data['worldwide_gross']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35, random_state=42)

logging.info('Setting Hyperparameter..........')

param_tunning = {'core_model__n_estimators': range(20,501,20)} 

grid_search= GridSearchCV(model,
                       param_grid = param_tunning,
                       scoring='r2',
                       cv=5) 

logging.info('Training model..........')

grid_search.fit(x_train, y_train)

logging.info('Cross validating..........')
final_results = cross_validate(grid_search.best_estimator_, x_train, y_train, return_train_score=True, cv=5 )

train_score = np.mean(final_results['train_score']) 
test_score  = np.mean(final_results['test_score']) 

assert train_score > 0.7
assert test_score > 0.65

logging.info(f'train_score {train_score}..........')

logging.info(f'test_score {test_score}..........')

update_model(grid_search.best_estimator_)

logging.info('Genera..........')
validation_score = grid_search.best_estimator_.score(x_test, y_test)
save_simple_metrics_report(train_score, test_score, validation_score, grid_search.best_estimator_)

y_test_pred = grid_search.best_estimator_.predict(x_test)
get_model_performance_set(y_test, y_test_pred)

logging.info('Done..........')