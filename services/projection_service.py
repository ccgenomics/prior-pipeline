import os
import pandas as pd
from sklearn.manifold import TSNE

class ProjectionService:
    @classmethod
    def project(cls, data_path, dims=[2], output=''):
        df = pd.read_csv(data_path, index_col = 0)
        base = os.path.basename(data_path)
        fname = os.path.splitext(base)[0]
        for n_components in dims:
            print("Training T-SNE for {}D...".format(n_components))
            X_embedded = TSNE(n_components=n_components, method='exact').fit_transform(df)
            X_embedded_df = pd.DataFrame(X_embedded, index=df.index)
            X_embedded_df.to_csv(output+'/'+fname+'_TSNE_{}D.csv'.format(n_components))
