# prior-pipeline

## Setup a worker

```bash
    git clone git@github.com:ccgenomics/prior-pipeline.git
    cd prior-pipeline
    python3 setup.py install
    aws configure
    hyperopt-mongo-worker --mongo=hy-mongo-server.co,:27017/hyperopt --poll-interval=0.1

```

## Usage

```python
import prior_pipeline as pp

from hyperopt import hp
from prior_pipeline.custom_models.my_mlp import FunnelMLPClassifier

mongo = {
            'url': 'mongo://{}:27017/hyperopt/jobs'.format(mongo_host),
            'exp': 'test_run_2'
        }

dims = [3, 5, 10, 15, 20, 40, 70]
data_paths = ['../data_s3/TCGA_Metabric_RNA_TSNE_{}.csv'.format(i) for i in dims]
targets_path =  '../data_s3/TCGA_Metabric_2_years.csv'

space = (hp.choice('model',
        [
            {
                'name': FunnelMLPClassifier,
                'params': {
                    'activation': hp.choice('activation', ['tanh', 'relu']),
                    'alpha': hp.choice('alpha', [5e-2]),
                    'solver' : hp.choice('solver', ['adam', 'lbfgs', 'sgd']),
                    'learning_rate': hp.choice('learning_rate', ['constant', 'adaptive']),
                    'input_layer': hp.quniform('input_layer', 2, 300, 1),
                    'output_layer': hp.quniform('output_layer', 2, 300, 1),
                    'num_hidden_layer': hp.quniform('num_hidden_layer', 2, 4, 1)
                }
            }
        ]),
        hp.choice('data_path', data_paths),
        targets_path)

best = pp.classifier.fit(space=space, max_evals=1000, mongo=mongo)
```

## Expected Data Form

The data and the targets are expected in two separate files with index.
(For classifier fit, target should be the class integer)

Such as:

    #data.csv

    id,col_1, col_2
    0,456, 678
    1,546, 878    
    2,456, 686    
    3,3656, 658    
    4,4357, 683    



    #tagets.csv

    id,target
    0,1
    1,1
    2,1
    3,0
    4,0

## Expected Model Interface

In order to use a custom model, the following methods need to be implemented:

```python
    class MyCustomModel:

      def fit(X, Y):
        pass

      def predict_proba(X):
        """
        Returns one class probabilities.

        Parameters:
        X (list): Input data of size N x X

        Returns:
        list: The one class probabilities vector of size N x 2 (eg: [[0.7,0.3],...])

        """
```
