from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.convolution_based import Rocket, MiniRocket, MultiRocket
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import polars as pl
import numpy as np
from autotsc.utils import load_dataset
import os
import time
from aeon.classification.convolution_based import RocketClassifier, MultiRocketClassifier, MiniRocketClassifier
import tensorflow as tf
from aeon.transformations.collection.interval_based import QUANTTransformer
from time import perf_counter
from sklearn.pipeline import make_pipeline
from aeon.classification.sklearn import SklearnClassifierWrapper
from tabpfn import TabPFNClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier, SummaryClassifier
from aeon.classification.deep_learning import LITETimeClassifier
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.distance_based import ProximityForest
from aeon.classification.deep_learning import IndividualLITEClassifier
from aeon.classification.interval_based import DrCIFClassifier
import ray
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.pipeline import make_pipeline as aeon_make_pipeline
from aeon.transformations.series.smoothing import MovingAverage

class Difference(BaseCollectionTransformer):
    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
    }

    def __init__(self, lag: int = 1) -> None:
        self.lag = lag
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X with shape (n_instances, n_channels, n_timesteps)."""
        if self.lag <= 0:
            raise ValueError(f"lag must be > 0, got {self.lag}")

        # X shape: (n_instances, n_channels, n_timesteps)
        # Apply difference along time axis (axis=2)
        Xt = X[:, :, self.lag:] - X[:, :, :-self.lag]
        return Xt


@ray.remote(num_cpus=2)
def train_fold(model_id, classifier, fold_id, X, y, folds):
    selected_fold = folds.filter(pl.col("fold") == fold_id).to_dicts()[0]
    train_idx = selected_fold['train_idx']
    test_idx = selected_fold['test_idx']
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]

    start_time = perf_counter()
    classifier.fit(X_train, y_train)
    end_time = perf_counter()
    training_time = end_time - start_time
    print(f"Model {repr(classifier).replace('\n', '').replace(' ', '')} Fold {fold_id} trained in {training_time:.2f} seconds")

    y_pred = classifier.predict(X_test)
    y_pred_zip = zip(test_idx, y_pred.tolist())
    return model_id, classifier, y_pred_zip

class AutoTSCModel2(BaseClassifier):
    # TODO: change capability tags
    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def set_use_models(self, model_ids):
        self.use_models = model_ids

    def __init__(self, n_jobs=-1, n_gpus=-1, n_folds=8, verbose=0):
        self.n_jobs = n_jobs
        self.n_gpus = n_gpus
        self.n_folds = n_folds
        self.verbose = verbose

        self.models_ = []
        self.summary_ = []

        # Each model uses 2 jobs, Ray has 8 CPUs, so 4 models can run concurrently
        model_n_jobs = 4

        model_creators = [
            #lambda: DrCIFClassifier(n_jobs=model_n_jobs, time_limit_in_minutes=1),
            lambda: FreshPRINCEClassifier(n_jobs=model_n_jobs, default_fc_parameters="minimal"),
            lambda: SummaryClassifier(),
            #lambda: aeon_make_pipeline(
            #    Difference(),
            #    SummaryClassifier()
            #),
            lambda: SklearnClassifierWrapper(
                make_pipeline(
                    StandardScaler(),
                    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                )
            ),
            #lambda: SklearnClassifierWrapper(
            #    TabPFNClassifier(n_preprocessing_jobs=model_n_jobs)
            #),
            lambda: SklearnClassifierWrapper(
                RandomForestClassifier(n_estimators=100, n_jobs=model_n_jobs)
            ),
            lambda: SklearnClassifierWrapper(
                RandomForestClassifier(n_estimators=100, n_jobs=model_n_jobs, ccp_alpha=0.01)
            ),
            lambda: SklearnClassifierWrapper(
                RandomForestClassifier(n_estimators=100, n_jobs=model_n_jobs, ccp_alpha=0.001)
            ),
            lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=0, n_kernels=2000),
            lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=0, n_kernels=2000),
            lambda: Catch22Classifier(n_jobs=model_n_jobs),
            lambda: aeon_make_pipeline(
                Difference(),
                Catch22Classifier()
            ),
            lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=1),
            lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=1),
            lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=20000),
            lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=20000),

            lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=50000),
            lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=50000),

            lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=100000),
            lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=100000),

            #lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=200000),
            #lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=200000),

            #lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=200000),
            #lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=200000),

            #lambda: MultiRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=200000),
            #lambda: MiniRocketClassifier(n_jobs=model_n_jobs, random_state=1, n_kernels=200000),

            #lambda: ProximityForest(n_jobs=model_n_jobs, max_depth=3, n_trees=5),
            # lambda: TemporalDictionaryEnsemble(n_jobs=model_n_jobs),
        ]

        self.models = []
        for creator in model_creators:
            try:
                model = creator()
                self.models.append(model)
            except Exception as e:
                model_name = str(e).split("'")[1] if "'" in str(e) else type(e).__name__
                print(f"Skipping model due to failed initialization: {model_name}")

        super().__init__()

    def build_metamodel(self):
        from sklearn.preprocessing import OneHotEncoder
        print("Building metamodel...")
        preds = []
        for fold_id in range(self.n_folds):
            selected_fold = self.folds_.filter(pl.col("fold") == fold_id).to_dicts()[0]
            train_idx = selected_fold['train_idx']
            test_idx = selected_fold['test_idx']

            X = []
            y = []
            for row in self.summary().iter_rows(named=True):
                X.append(row['fold_predictions'])
                y = row['true_labels']
            X = np.array(X).T
            y = np.array(y)

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]

            pipeline = Pipeline([
                ('encoder', OneHotEncoder(sparse_output=False)),
                ('classifier', RidgeClassifierCV())
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            preds.extend(zip(test_idx, y_pred.tolist()))
        fold_predictions = sorted(preds)
        fold_predictions = [p[1] for p in fold_predictions]
        acc = accuracy_score(self.y_, fold_predictions)
        self.summary_.append({
            'model_id': self.model_id,
            'classifier': repr(pipeline).replace('\n', '').replace(' ', ''),
            'fold_predictions': fold_predictions,
            'true_labels': self.y_.tolist(),
            'validation_accuracy': acc,
        })
        self.model_id += 1

    def build_metamodel2(self):
        from sklearn.preprocessing import OneHotEncoder
        print("Building metamodel...")
        preds = []
        for fold_id in range(self.n_folds):
            selected_fold = self.folds_.filter(pl.col("fold") == fold_id).to_dicts()[0]
            train_idx = selected_fold['train_idx']
            test_idx = selected_fold['test_idx']

            X = []
            y = []
            for row in self.summary().iter_rows(named=True):
                X.append(row['fold_predictions'])
                y = row['true_labels']
            X = np.array(X).T
            y = np.array(y)

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]

            pipeline = RandomForestClassifier(n_estimators=300, n_jobs=self.n_jobs)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            preds.extend(zip(test_idx, y_pred.tolist()))
        fold_predictions = sorted(preds)
        fold_predictions = [p[1] for p in fold_predictions]
        acc = accuracy_score(self.y_, fold_predictions)
        self.summary_.append({
            'model_id': self.model_id,
            'classifier': repr(pipeline).replace('\n', '').replace(' ', ''),
            'fold_predictions': fold_predictions,
            'true_labels': self.y_.tolist(),
            'validation_accuracy': acc,
        })
        self.model_id += 1


    def _fit(self, X, y):
        ray.init(num_cpus=24, ignore_reinit_error=True, num_gpus=0)


        if self.verbose > 0:
            # print Y, y statistics
            print('Datase shape:', X.shape)
            print('Number of classes:', len(np.unique(y)))

            n_cpus_available = os.cpu_count() or 1
            n_cpus_to_use = n_cpus_available if self.n_jobs == -1 else self.n_jobs
            print(f"CPUs: {n_cpus_to_use}/{n_cpus_available}")

            n_gpus_available = len(tf.config.list_physical_devices('GPU'))
            n_gpus_to_use = n_gpus_available if self.n_jobs == -1 else min(self.n_jobs, n_gpus_available)
            print(f"GPUs: {n_gpus_to_use}/{n_gpus_available}")

        self.X_ = X
        self.y_ = y
        self.folds_ = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            self.folds_.append({
                'fold': i,
                'train_idx': train_idx,
                'test_idx': test_idx,
            })
        self.folds_ = pl.DataFrame(self.folds_)
        
        tasks = []
        self.model_id = 0
        model_training_start_time = perf_counter()
        for model_ in self.models:
            for fold in range(self.n_folds):
                task = train_fold.remote(
                    self.model_id,
                    clone(model_),
                    fold,
                    self.X_,
                    self.y_,
                    self.folds_
                )
                tasks.append(task)
            self.model_id += 1

        model_predictions = {}
        model_classfiers = {}

        results = ray.get(tasks)
        model_training_end_time = perf_counter()
        training_duration = model_training_end_time - model_training_start_time
        print(f"Trained models in {training_duration:.2f} seconds")

        for model_id, model, y_pred in results:
            if model_id not in model_predictions:
                model_predictions[model_id] = []
            model_predictions[model_id].extend(y_pred)

            if model_id not in model_classfiers:
                model_classfiers[model_id] = []
            model_classfiers[model_id].append(model)

        for model_id in model_predictions:
            fold_predictions = sorted(model_predictions[model_id])
            fold_predictions = [p[1] for p in fold_predictions]
            self.models_.append(tuple(model_classfiers[model_id]))
            acc = accuracy_score(self.y_, fold_predictions)

            self.summary_.append({
                'model_id': model_id,
                'classifier': repr(self.models[model_id]).replace('\n', '').replace(' ', ''),
                'fold_predictions': fold_predictions,
                'true_labels': self.y_.tolist(),
                'validation_accuracy': acc,
            })

        self.build_metamodel()
        self.build_metamodel2()

        ray.shutdown() 
        return self

    def get_avaiable_models(self):
        return self.summary()['model_id'].to_list()

    def summary(self):
        return pl.DataFrame(self.summary_).sort('validation_accuracy')
    
    def most_common_label(self, all_predictions):
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[counts.argmax()])
        return np.array(final_predictions)

    def _predict(self, X):
        if len(self.models_) == 0:
            raise ValueError("No models trained yet. Call fit().")

        all_predictions = []
        for models in self.models_:
            for model in models:
                predictions = model.predict(X)
                all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)
        return self.most_common_label(all_predictions)
    
class AutoTSCModel(BaseClassifier):

    # TODO: change capability tags
    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(self, n_jobs=-1, n_gpus=-1, n_folds=8, verbose=0):
        self.n_jobs = n_jobs
        self.n_gpus = n_gpus
        self.step_counter_ = 0
        self.X_ = None
        self.y_ = None
        self.X_features_ = None
        self.feature_transformers_ = []
        self.models_ = []
        self.n_folds = n_folds
        self.verbose = verbose
        self.summary_ = []
        super().__init__()

    def _fit(self, X, y):
        if self.verbose > 0:
            n_cpus_available = os.cpu_count() or 1
            n_cpus_to_use = n_cpus_available if self.n_jobs == -1 else self.n_jobs
            print(f"CPUs: {n_cpus_to_use}/{n_cpus_available}")

            n_gpus_available = len(tf.config.list_physical_devices('GPU'))
            n_gpus_to_use = n_gpus_available if self.n_jobs == -1 else min(self.n_jobs, n_gpus_available)
            print(f"GPUs: {n_gpus_to_use}/{n_gpus_available}")

        self.X_ = X
        self.y_ = y
        X_features = X.reshape(X.shape[0], -1)
        self.X_features_ = pl.DataFrame(X_features, schema=[f"raw|step_{i}" for i in range(X_features.shape[1])])

        self.folds_ = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            self.folds_.append({
                'fold': i,
                'train_idx': train_idx,
                'test_idx': test_idx,
            })
        self.folds_ = pl.DataFrame(self.folds_)

        if self.verbose > 0:
            print(f"Created {len(self.folds_)} stratified folds for training.")
        return self
    
    def step(self):
        self.add_random_features()

        feature_subset = self.select_random_features()
        classifier = self.get_random_tabular_model()

        model_training_start_time = time.time()
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._train_fold)(
                clone(classifier),
                fold,
                feature_subset
            )
            for fold in range(self.n_folds)
        )
        model_training_end_time = time.time()
        training_duration = model_training_end_time - model_training_start_time
        print(f"Trained models for {self.n_folds} folds in {training_duration:.2f} seconds")

        fold_predictions = []
        fold_models = []
        for model, y_pred in results:
            y_pred = list(y_pred)
            fold_models.append(model)
            fold_predictions.extend(y_pred)

        fold_predictions = sorted(fold_predictions)
        fold_predictions = [p[1] for p in fold_predictions]
        self.models_.append(tuple(fold_models))
        acc = accuracy_score(self.y_, fold_predictions)

        self.summary_.append({
            'step': self.step_counter_,
            'classifier': repr(classifier),
            'training_time_seconds': training_duration,
            'fold_predictions': fold_predictions,
            'true_labels': self.y_.tolist(),
            'validation_accuracy': acc,
        })
        self.step_counter_ += 1

    def _train_fold(self, classifier, fold_id, feature_subset):
        selected_fold = self.folds_.filter(pl.col("fold") == fold_id).to_dicts()[0]
        train_idx = selected_fold['train_idx']
        test_idx = selected_fold['test_idx']
        
        features_train = self.X_features_.select(feature_subset)
        features_test = self.X_features_.select(feature_subset)

        features_train = features_train.with_row_index("idx").filter(pl.col("idx").is_in(train_idx)).drop("idx")
        features_test = features_test.with_row_index("idx").filter(pl.col("idx").is_in(test_idx)).drop("idx")

        y_train = self.y_[train_idx]

        classifier.fit(features_train, y_train)

        y_pred = classifier.predict(features_test)
        y_pred_zip = zip(test_idx, y_pred.tolist())
        return classifier, y_pred_zip

    def get_random_tabular_model(self):
        if np.random.rand() < 0.5:
            alphas = np.logspace(-4, 4, np.random.randint(9, 16))
            classifier = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('classifier', RidgeClassifierCV(alphas=alphas))
            ])
            return classifier
        else:
            ccp_alphas = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
            selected = np.random.choice(ccp_alphas)
            classifier = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, ccp_alpha=selected)
            return classifier
        
    def get_random_feature_generator(self):
        rocket_types = [
            Rocket(n_kernels=500, n_jobs=self.n_jobs, random_state=None),
            MiniRocket(n_kernels=500, n_jobs=self.n_jobs, random_state=None),
            MultiRocket(n_jobs=self.n_jobs),
            QUANTTransformer(),
            #Catch22Transformer(),
        ]
        random_index = np.random.randint(len(rocket_types))
        return rocket_types[random_index]

    def add_random_features(self):
        generator = self.get_random_feature_generator()
        generator_name = generator.__class__.__name__.lower()

        start_time = time.time()
        new_features = generator.fit_transform(self.X_)
        feature_extraction_time = time.time() - start_time
        self.feature_transformers_.append(generator)

        start_idx = self.X_features_.shape[1]
        feature_cols = [f"{generator_name}|feat_{start_idx + i}" for i in range(new_features.shape[1])]
        new_features_df = pl.DataFrame(new_features, schema=feature_cols)
        self.X_features_ = pl.concat([self.X_features_, new_features_df], how="horizontal")

        if self.verbose > 0:
            print(f"Generated {new_features.shape[1]} {generator_name} features in {feature_extraction_time:.2f}s")

    def select_random_features(self, max_features=20000):
        n_features = self.X_features_.shape[1]
        all_feature_names = self.X_features_.columns

        if n_features > max_features:
            rng = np.random.RandomState(None)
            feature_indices = rng.choice(n_features, max_features, replace=False)
            selected_features = [all_feature_names[i] for i in feature_indices]
        else:
            selected_features = all_feature_names

        return selected_features

    def _predict(self, X):
        if len(self.models_) == 0:
            raise ValueError("No models trained yet. Call fit()/step().")

        # Start with raw features
        X_features_flat = X.reshape(X.shape[0], -1)
        X_features = pl.DataFrame(X_features_flat, schema=[f"raw|step_{i}" for i in range(X_features_flat.shape[1])])

        # Add transformed features with same naming convention as in add_random_features
        feature_dfs = [X_features]
        start_idx = X_features.shape[1]

        for rocket in self.feature_transformers_:
            generator_name = rocket.__class__.__name__.lower()
            new_features = rocket.transform(X)

            feature_cols = [f"{generator_name}|feat_{start_idx + i}" for i in range(new_features.shape[1])]
            new_features_df = pl.DataFrame(new_features, schema=feature_cols)
            feature_dfs.append(new_features_df)
            start_idx += new_features.shape[1]

        X_features = pl.concat(feature_dfs, how="horizontal")
    
        models_to_use = self.models_
        models_to_use = [m for i, m in enumerate(self.models_) if i in self.best_models]

        all_predictions = []
        for model_group in models_to_use:
            for model in model_group:
                features = X_features.select(model.feature_names_in_)
                predictions = model.predict(features)
                all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)
        return self.most_common_label(all_predictions)

    def most_common_label(self, all_predictions):
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[counts.argmax()])
        return np.array(final_predictions)

    def summary(self):
        return pl.DataFrame(self.summary_)
    
    def build_ensemble(self):
        top_3_models = pl.DataFrame(self.summary_).sort("validation_accuracy").tail(len(self.summary_) // 3)
        self.best_models = top_3_models['step'].to_list()
        print(f"selected models ({len(self.best_models)}/{len(self.summary_)}): {self.best_models}")
        pass


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset("ArrowHead")
    n_jobs = -1
    model = AutoTSCModel(verbose=1, n_jobs=n_jobs)
    model.fit(X_train, y_train)

    for _ in range(18):
        model.step()

    model.build_ensemble()

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"AutoTSCModel Test accuracy: {acc}")
    print(model.summary())

    model = MultiRocketClassifier(n_jobs=n_jobs)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"RocketClassifier Test accuracy: {acc}")