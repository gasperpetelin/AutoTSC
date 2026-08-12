"""WEASEL v2 dictionary features."""

import numpy as np
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2


class WEASELTransformerV2Unsupervised(WEASELTransformerV2):
    """WEASELTransformerV2 usable in this project's unsupervised feature pipeline.

    Upstream always does ``y.copy()`` in fit_transform, but feature_selection="none"
    means y is otherwise unused, so a placeholder y stands in for the real labels
    that the shared feature-fitting call sites don't have access to.
    """

    def __init__(
        self,
        min_window=4,
        norm_options=(False,),
        word_lengths=(7, 8),
        use_first_differences=(True, False),
        max_feature_count=30_000,
        random_state=None,
        n_jobs=4,
    ):
        super().__init__(
            min_window=min_window,
            norm_options=norm_options,
            word_lengths=word_lengths,
            use_first_differences=use_first_differences,
            feature_selection="none",
            max_feature_count=max_feature_count,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit_transform(self, X, y=None):
        if y is None:
            y = np.zeros(X.shape[0], dtype=int)

        self.transformers_ = []
        Xt = []
        for channel in range(X.shape[1]):
            transformer = WEASELTransformerV2(
                min_window=self.min_window,
                norm_options=self.norm_options,
                word_lengths=self.word_lengths,
                use_first_differences=self.use_first_differences,
                feature_selection="none",
                max_feature_count=self.max_feature_count,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            Xt.append(transformer.fit_transform(X[:, channel : channel + 1], y))
            self.transformers_.append(transformer)
        return np.hstack(Xt)

    def transform(self, X, y=None):
        if X.shape[1] != len(self.transformers_):
            raise ValueError("X must have the same number of channels as the training data")
        return np.hstack(
            [
                transformer.transform(X[:, channel : channel + 1])
                for channel, transformer in enumerate(self.transformers_)
            ]
        )

    def _transform(self, X, y=None):
        return super()._transform(np.asarray(X, dtype=np.float64), y)
