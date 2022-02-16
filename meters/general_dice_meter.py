import typing as t
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from torch import Tensor

from meters import Metric
from utils.general import to_float, class2one_hot, probs2one_hot, one_hot, simplex, average_iter

metric_result = t.Dict[str, t.Union[float, np.ndarray, Tensor]]


class UniversalDice(Metric[metric_result]):
    def __init__(self, C: int, report_axis: t.Iterable[int] = None) -> None:
        super(UniversalDice, self).__init__()
        if report_axis:
            assert max(list(report_axis)) <= C, (
                "Incompatible parameter of `C`={} and "
                "`report_axises`={}".format(C, report_axis)
            )
        self._C = C
        self._report_axis = list(range(self._C))
        if report_axis is not None:
            self._report_axis = list(report_axis)
        self.reset()

    def reset(self):
        self._intersections: t.DefaultDict[str, Tensor] = defaultdict(lambda: 0)  # noqa
        self._unions: t.DefaultDict[str, Tensor] = defaultdict(lambda: 0)  # noqa
        self._n = 0

    @torch.no_grad()
    def _add(self, pred: Tensor, target: Tensor, *, group_name: t.Union[str, t.List[str]] = None):  # noqa
        """
        add pred and target
        :param pred: class- or onehot-coded tensor of the same shape as the target
        :param target: class- or onehot-coded tensor of the same shape as the pred
        :param group_name: List of names, or a string of a name, or None.
                        indicating 2D slice dice, batch-based dice
        :return:
        """

        assert pred.shape == target.shape, (
            f"incompatible shape of `pred` and `target`, given "
            f"{pred.shape} and {target.shape}."
        )
        pred, target = pred.detach(), target.detach()
        onehot_pred, onehot_target = self._convert2onehot(pred, target)
        B, C, *hw = pred.shape

        if group_name is None:
            group_name = [str(self._n) + f"_{i:03d}" for i in range(B)]  # make it like slice based dice
        else:
            if isinstance(group_name, str):
                group_name = [group_name for _ in range(B)]
            elif isinstance(group_name, (tuple, list)):
                group_name = list(group_name)
            else:
                raise TypeError(f"type of `group_name` wrong {type(group_name)}")
        assert len(group_name) == B

        interaction, union = (
            self._intersection(onehot_pred, onehot_target),
            self._union(onehot_pred, onehot_target),
        )
        for _int, _uni, g in zip(interaction, union, group_name):
            self._intersections[g] += _int
            self._unions[g] += _uni
        self._n += 1

    def compute_dice_by_group(self) -> t.Optional[Tensor]:
        if self._n > 0:
            dices = self._compute_dice(intersection=torch.stack(tuple(self._intersections.values()), dim=0),
                                       union=torch.stack(tuple(self._unions.values()), dim=0))
            return dices

    @staticmethod
    def _compute_dice(intersection: Tensor, union: Tensor) -> Tensor:
        return (2 * intersection.float() + 1e-16) / (union.float() + 1e-16)

    def _summary(self) -> metric_result:
        if self._n > 0:
            dices = self.compute_dice_by_group()
            means, stds = dices.mean(dim=0), dices.std(dim=0)
        else:
            means, stds = (np.nan,) * self._C, (np.nan,) * self._C
        report_dict = {f"DSC{i}": to_float(means[i]) for i in self._report_axis}
        report_dict.update({"DSC_mean": average_iter(report_dict.values())})
        return report_dict

    @property
    def group_names(self):
        return sorted(self._intersections.keys())

    @staticmethod
    def _intersection(pred: Tensor, target: Tensor):
        """
        return the interaction, supposing the two inputs are onehot-coded.
        :param pred: onehot pred
        :param target: onehot target
        :return: tensor of intersaction over classes
        """
        intersect = (pred * target).sum(list(range(2, pred.dim()))).long()
        return intersect

    @staticmethod
    def _union(pred: Tensor, target: Tensor):
        """
        return the union, supposing the two inputs are onehot-coded.
        :param pred: onehot pred
        :param target: onehot target
        :return: tensor of intersaction over classes
        """
        union = (pred + target).sum(list(range(2, pred.dim()))).long()
        return union

    def _convert2onehot(self, pred: Tensor, target: Tensor) -> t.Tuple[Tensor, Tensor]:
        # only two possibility: both onehot or both class-coded.
        # if they are onehot-coded:
        if simplex(pred, 1) and one_hot(target):
            return probs2one_hot(pred).long(), target.long()
        # here the pred and target are labeled long
        return (
            class2one_hot(pred, self._C).long(),
            class2one_hot(target, self._C).long(),
        )

    def __repr__(self):
        string = f"C={self._C}, report_axis={self._report_axis}\n"
        return string + "\t" + str(self.summary())


if __name__ == '__main__':
    meter = UniversalDice(C=4, report_axis=[1, 2])
    pred = torch.randn(10, 4, 225, 225, device="cuda").max(1)[1]
    target = torch.randn(10, 4, 225, 225, device="cuda").max(1)[1]
    meter.add(pred, target)

    result = meter.summary()
    pprint(result)
