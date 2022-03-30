import os.path
import random
import re
from collections import Counter
from collections import defaultdict
from copy import deepcopy as dcopy
from typing import List

from torch.utils.data import Sampler


class ContrastBatchSampler(Sampler[List[int]]):
    """
    This class is going to realize the sampling for different patients and from the same patients
    `we form batches by first randomly sampling m < M volumes. Then, for each sampled volume, we sample one image per
    partition resulting in S images per volume. Next, we apply a pair of random transformations on each sampled image and
    add them to the batch
    """

    class _SamplerIterator:

        def __init__(self, scan2index, partition2index, scan_sample_num=4, partition_sample_num=1,
                     shuffle=False) -> None:
            self._group2index, self._partition2index = dcopy(scan2index), dcopy(partition2index)

            assert 1 <= scan_sample_num <= len(self._group2index.keys()), scan_sample_num
            self._scan_sample_num = scan_sample_num
            self._partition_sample_num = partition_sample_num
            self._shuffle = shuffle

        def __iter__(self):
            return self

        def __next__(self):
            batch_index = []
            cur_group_samples = random.sample(list(self._group2index.keys()), self._scan_sample_num)
            assert isinstance(cur_group_samples, list), cur_group_samples

            # # for each group sample, choose at most partition_sample_num slices per partition
            # for cur_group in cur_group_samples:
            #     available_slices_given_group = self._group2index[cur_group]
            #     for s_available_slices in self._partition2index.values():
            #         try:
            #             sampled_slices = random.sample(
            #                 sorted(set(available_slices_given_group) & set(s_available_slices)),
            #                 self._partition_sample_num
            #             )
            #             batch_index.extend(sampled_slices)
            #         except ValueError:
            #             return self.__next__()
            # if self._shuffle:
            #     random.shuffle(batch_index)
            # return batch_index

            fixed_one_partition = random.sample(self._partition2index.keys(), k=1)[0]

            # for each group sample, choose at most partition_sample_num slices per partition
            for cur_group in cur_group_samples:
                available_slices_given_group = self._group2index[cur_group]
                # for s_available_slices in fixed_one_partition :
                s_available_slices = self._partition2index[fixed_one_partition]
                try:
                    sampled_slices = random.sample(
                        sorted(set(available_slices_given_group) & set(s_available_slices)),
                        self._partition_sample_num
                    )
                    batch_index.extend(sampled_slices)
                except ValueError:
                    return self.__next__()
            if self._shuffle:
                random.shuffle(batch_index)
            return batch_index

    def __init__(self, dataset, scan_sample_num=4, partition_sample_num=1, shuffle=False) -> None:
        super(ContrastBatchSampler, self).__init__(data_source=dataset)
        self._dataset = dataset
        filenames = dcopy(dataset.get_filenames())
        scan2index, partition2index = defaultdict(lambda: []), defaultdict(lambda: [])

        def _get_scan_name(filename) -> str:
            return re.compile(r"\d+").findall(os.path.basename(filename))[0]

        # get total slice_num per scan
        total_scan_dict = Counter([_get_scan_name(filename) for filename in filenames])
        total_scan_dict = dict(sorted(total_scan_dict.items()))

        def _get_partition(filename):
            scan_name = _get_scan_name(filename)
            slice_num = int(re.compile(r"\d+").findall(os.path.basename(filename))[1])
            total_scan_num = total_scan_dict[scan_name]
            return min(slice_num // (total_scan_num // dataset.partition_num), dataset.partition_num-1)

        for i, filename in enumerate(filenames):
            group = _get_scan_name(filename)  # noqa
            scan2index[group].append(i)  # noqa
            partition = _get_partition(filename)  # noqa
            partition2index[partition].append(i)  # noqa

        self._scan2index = scan2index
        self._partition2index = partition2index
        self._scan_sample_num = scan_sample_num
        self._partition_sample_num = partition_sample_num
        self._shuffle = shuffle

    def __iter__(self):
        return self._SamplerIterator(self._scan2index, self._partition2index, self._scan_sample_num,
                                     self._partition_sample_num, shuffle=self._shuffle)

    def __len__(self) -> int:
        return len(self._dataset)
