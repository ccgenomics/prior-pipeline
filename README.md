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
