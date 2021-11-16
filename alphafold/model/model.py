# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for constructing the model."""
from typing import Any, Mapping, Optional, Union, List, Tuple

from absl import logging
from alphafold.common import confidence
from alphafold.model import features
from alphafold.model import modules
import haiku as hk
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf
import tree


def get_confidence_metrics(
    prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
  """Post processes prediction_result to get confidence metrics."""

  confidence_metrics = {}
  confidence_metrics['plddt'] = confidence.compute_plddt(
      prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(confidence.compute_predicted_aligned_error(
        prediction_result['predicted_aligned_error']['logits'],
        prediction_result['predicted_aligned_error']['breaks']))
    confidence_metrics['ptm'] = confidence.predicted_tm_score(
        prediction_result['predicted_aligned_error']['logits'],
        prediction_result['predicted_aligned_error']['breaks'])

  return confidence_metrics


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None):
    self.config = config
    self.params = params

    def _forward_fn(batch):
      model = modules.AlphaFold(self.config.model)
      return model(
          batch,
          is_training=False,
          compute_loss=False,
          ensemble_representations=True)

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init = jax.jit(hk.transform(_forward_fn).init)

  def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
    """Initializes the model parameters.

    If none were provided when this class was instantiated then the parameters
    are randomly initialized.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: A random seed to use to initialize the parameters if none
        were set when this class was initialized.
    """
    if not self.params:
      # Init params randomly.
      rng = jax.random.PRNGKey(random_seed)
      self.params = hk.data_structures.to_mutable_dict(
          self.init(rng, feat))
      logging.warning('Initialized parameters randomly')

  def process_features(
      self,
      raw_features: Union[tf.train.Example, features.FeatureDict],
      random_seed: int) -> features.FeatureDict:
    """Processes features to prepare for feeding them into the model.

    Args:
      raw_features: The output of the data pipeline either as a dict of NumPy
        arrays or as a tf.train.Example.
      random_seed: The random seed to use when processing the features.

    Returns:
      A dict of NumPy feature arrays suitable for feeding into the model.
    """
    if isinstance(raw_features, dict):
      return features.np_example_to_features(
          np_example=raw_features,
          config=self.config,
          random_seed=random_seed)
    else:
      return features.tf_example_to_features(
          tf_example=raw_features,
          config=self.config,
          random_seed=random_seed)

  def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
    self.init_params(feat)
    logging.info('Running eval_shape with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
    logging.info('Output shape was %s', shape)
    return shape

  def predict(self, feat: features.FeatureDict, ret_all_cycle: bool) -> List[Tuple[Mapping[str, Any], int]]:
    """Makes a prediction by inferencing the model on the provided features.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.

    Returns:
      A List of Tuple of a dictionary of model outputs and the recycle number.
    """
    self.init_params(feat)
    logging.info('Running predict with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    result, recycles, result_tuple = self.apply(self.params, jax.random.PRNGKey(0), feat)
    # This block is to ensure benchmark timings are accurate. Some blocking is
    # already happening when computing get_confidence_metrics, and this ensures
    # all outputs are blocked on.

    def iter_result(result):
      result_ = result[0]
      jax.tree_map(lambda x: x.block_until_ready(), result_)
      result_.update(get_confidence_metrics(result_))
      logging.info('Output shape was %s',
                   tree.map_structure(lambda x: x.shape, result_))
      return result

    if not ret_all_cycle:  # Return last cycles
      ret = iter_result((result, recycles))
      return ret
    else:  # Return all cycles
      def get_each_result(i):
        recycles = result_tuple[0][i]
        result = {key1: {key2: val2[i + 1] for key2, val2 in val1.items()} for key1, val1 in result_tuple[1].items()}
        return result, recycles

      result_list = [get_each_result(i) for i in range(recycles - 1)]
      result_list.append(result)
      ret_list = [iter_result(result) for result in result_list]
      return ret_list
