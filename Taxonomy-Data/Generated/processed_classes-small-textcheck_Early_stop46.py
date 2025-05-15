import numpy as np
import logging

class EarlyStopping:
    def __init__(self, metric, min_delta=1e-14, patience=5, threshold=0.0):
        self._validate_arguments(metric, min_delta, patience, threshold)
        self._dtype = {
            'names': ['epoch', 'count', 'train_acc', 'train_loss', 'val_acc', 'val_loss'],
            'formats': [int, int, float, float, float, float]
        }
        self.metric = metric
        self.min_delta = min_delta
        self.patience = patience
        self.threshold = threshold
        self._index_best = -1
        self._history = np.empty((0,), dtype=self._dtype)

    def _validate_arguments(self, metric, min_delta, patience, threshold):
        if min_delta < 0:
            raise ValueError('Invalid value encountered: "min_delta" needs to be greater than zero.')
        if patience < 0 and threshold <= 0:
            raise ValueError('Invalid configuration encountered: Either "patience" or "threshold" must be enabled.')
        if '_acc' in metric.name and (threshold < 0.0 or threshold > 1.0):
            raise ValueError('Invalid value encountered: "threshold" needs to be within the interval [0, 1] for accuracy metrics.')

    def check_early_stop(self, epoch, measured_values):
        # Validate epoch
        if epoch < 0:
            raise ValueError("Epoch number must be non-negative.")
        
        # Validate measured_values
        if self.metric.name not in measured_values:
            raise ValueError(f"Metric '{self.metric.name}' not found in measured values.")
        
        current_value = measured_values[self.metric.name]
        
        # Check if the metric has crossed the threshold
        if self.metric.better == 'higher' and current_value >= self.threshold:
            logging.debug(f"Early stopping: Metric {self.metric.name} has crossed the threshold {self.threshold}.")
            return True
        elif self.metric.better == 'lower' and current_value <= self.threshold:
            logging.debug(f"Early stopping: Metric {self.metric.name} has crossed the threshold {self.threshold}.")
            return True
        
        # Add current measurements to history
        self.add_to_history(epoch, measured_values)
        
        # Determine the sign for improvement
        metric_sign = 1 if self.metric.better == 'higher' else -1
        
        # Check for improvement
        if self._check_for_improvement(measured_values, metric_sign):
            return True
        
        return False

    def _check_for_improvement(self, measured_values, metric_sign):
        previous_best = self._history[self.metric.name][self._index_best]
        index_last = self._history.shape[0] - 1
        delta = measured_values[self.metric.name] - previous_best
        delta_sign = np.sign(delta)
        if self.min_delta > 0:
            improvement = delta_sign == metric_sign and np.abs(delta) >= self.min_delta
        else:
            improvement = delta_sign == metric_sign
        if improvement:
            self._index_best = index_last
            return False
        else:
            history_since_previous_best = self._history[self._index_best + 1:][self.metric.name]
            rows_not_nan = np.logical_not(np.isnan(history_since_previous_best))
            if rows_not_nan.sum() > self.patience:
                logging.debug(f'Early stopping: Patience exceeded.{{value={index_last-self._index_best}, patience={self.patience}}}')
                return True
            return False

    def add_to_history(self, epoch, measured_values):
        # Assume this method adds the current measurements to the history.
        # This is a placeholder implementation.
        new_entry = (epoch, 0, measured_values.get('train_acc', np.nan), measured_values.get('train_loss', np.nan),
                     measured_values.get('val_acc', np.nan), measured_values.get('val_loss', np.nan))
        self._history = np.append(self._history, np.array(new_entry, dtype=self._dtype))

