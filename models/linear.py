from models.nonlinear import MLP
import torch.nn as nn


class LinearModel(MLP):
    # noinspection PyAttributeOutsideInit
    def build_networks(self):
        self._mlp_t_w = nn.Linear(self.dim_w, self.dim_t * self.treatment_distribution.num_params)
        self._mlp_y0_w = nn.Linear(self.dim_w, self.dim_y * self.outcome_distribution.num_params)
        self._mlp_y1_w = nn.Linear(self.dim_w, self.dim_y * self.outcome_distribution.num_params)

        self.networks = [self._mlp_t_w, self._mlp_y0_w, self._mlp_y1_w]

    def mlp_t_w(self, w,  **kwargs):
        return self._mlp_t_w(w)

    def mlp_y_tw(self, wt, **kwargs):
        w, t = wt[:, :-1], wt[:, -1:]
        y0 = self._mlp_y0_w(w)
        y1 = self._mlp_y1_w(w)
        return y0 * (1 - t) + y1 * t
