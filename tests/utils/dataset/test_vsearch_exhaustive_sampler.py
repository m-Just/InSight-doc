from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.experimental.dataset.vsearch_sampler import VSearchExhaustiveBatchSampler, VSearchWeightedRandomRefillBatchSampler
from verl.trainer.main_ppo import create_rl_sampler


class MockDataFrame:
    def __init__(self, data_sources):
        self._data_sources = list(data_sources)
        self.column_names = ["data_source"]

    def __getitem__(self, key):
        if key != "data_source":
            raise KeyError(key)
        return self._data_sources


class MockRLHFDataset(Dataset):
    def __init__(self, data_sources):
        self.dataframe = MockDataFrame(data_sources)

    def __len__(self):
        return len(self.dataframe["data_source"])

    def __getitem__(self, idx):
        return {"idx": idx, "source": self.dataframe["data_source"][idx]}


def _config():
    return OmegaConf.create(
        {
            "seed": 7,
            "batch_sampler": {
                "weights": {
                    "source_a": 1.0,
                    "source_b": 1.0,
                }
            },
        }
    )


def _config_three_sources():
    return OmegaConf.create(
        {
            "seed": 11,
            "batch_sampler": {
                "weights": {
                    "source_a": 1.0,
                    "source_b": 1.0,
                    "source_c": 1.0,
                }
            },
        }
    )


def _config_from_weights(weights, seed=17):
    return OmegaConf.create(
        {
            "seed": seed,
            "batch_sampler": {
                "weights": weights,
            },
        }
    )


def _refill_config_from_weights(weights, seed=23):
    return OmegaConf.create(
        {
            "seed": seed,
            "batch_sampler": {
                "weights": weights,
                "stop_after": "max_source_exhaustion",
            },
        }
    )


def _config_from_weights_file(weights_file, seed=19, inline_weights=None):
    batch_sampler = {
        "weights_file": str(weights_file),
    }
    if inline_weights is not None:
        batch_sampler["weights"] = inline_weights
    return OmegaConf.create(
        {
            "seed": seed,
            "batch_sampler": batch_sampler,
        }
    )


def _create_rl_sampler_config():
    return OmegaConf.create(
        {
            "seed": 13,
            "shuffle": True,
            "train_batch_size": 4,
            "gen_batch_size": 4,
            "dataloader_num_workers": 8,
            "sampler": None,
            "batch_sampler": {
                "enabled": True,
                "class_path": "pkg://verl.experimental.dataset.vsearch_sampler",
                "class_name": "VSearchExhaustiveBatchSampler",
                "weights": {
                    "source_a": 1.0,
                    "source_b": 1.0,
                },
            },
        }
    )


def _sources_for_batch(batch, data_sources):
    return [data_sources[idx] for idx in batch]


def _flatten(batches):
    return [idx for batch in batches for idx in batch]


def _batch_source_counts(batch, data_sources):
    sources = _sources_for_batch(batch, data_sources)
    return {source: sources.count(source) for source in sorted(set(data_sources))}


def _normalize_loader_batch(batch):
    return {
        "idx": [int(idx) for idx in batch["idx"]],
        "source": list(batch["source"]),
    }


def test_exhaustive_sampler_continues_after_one_source_exhausts():
    data_sources = ["source_a"] * 2 + ["source_b"] * 10
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchExhaustiveBatchSampler(batch_size=4, data_source=dataset, data_config=_config())

    batches = list(sampler)

    assert len(sampler) == 3
    assert len(batches) == 3
    assert all(len(batch) == 4 for batch in batches)

    used_indices = [idx for batch in batches for idx in batch]
    assert sorted(used_indices) == list(range(12))

    first_batch_sources = _sources_for_batch(batches[0], data_sources)
    assert first_batch_sources.count("source_a") == 2
    assert first_batch_sources.count("source_b") == 2

    later_batch_sources = _sources_for_batch(batches[1] + batches[2], data_sources)
    assert later_batch_sources.count("source_a") == 0
    assert later_batch_sources.count("source_b") == 8


def test_exhaustive_sampler_drops_only_global_remainder():
    data_sources = ["source_a"] * 3 + ["source_b"] * 7
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchExhaustiveBatchSampler(batch_size=4, data_source=dataset, data_config=_config())

    batches = list(sampler)

    assert len(sampler) == 2
    assert len(batches) == 2
    assert all(len(batch) == 4 for batch in batches)

    used_indices = _flatten(batches)
    assert len(used_indices) == len(set(used_indices))
    assert len(used_indices) == (len(data_sources) // 4) * 4

    used_sources = _sources_for_batch(used_indices, data_sources)
    assert used_sources.count("source_a") == 3
    assert used_sources.count("source_b") == 5


def test_exhaustive_sampler_respects_weights_before_and_after_exhaustion():
    data_sources = ["source_a"] * 5 + ["source_b"] * 9 + ["source_c"] * 20
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchExhaustiveBatchSampler(
        batch_size=10,
        data_source=dataset,
        data_config=_config_from_weights({"source_a": 1.0, "source_b": 3.0, "source_c": 6.0}),
    )

    batches = list(sampler)

    assert len(sampler) == 3
    assert len(batches) == 3
    assert all(len(batch) == 10 for batch in batches)
    assert [_batch_source_counts(batch, data_sources) for batch in batches] == [
        {"source_a": 1, "source_b": 3, "source_c": 6},
        {"source_a": 1, "source_b": 3, "source_c": 6},
        {"source_a": 1, "source_b": 3, "source_c": 6},
    ]

    used_indices = _flatten(batches)
    assert len(used_indices) == len(set(used_indices))
    used_sources = _sources_for_batch(used_indices, data_sources)
    assert used_sources.count("source_a") == 3
    assert used_sources.count("source_b") == 9
    assert used_sources.count("source_c") == 18


def test_weighted_random_refill_sampler_refills_small_sources_until_all_sources_exhaust_once():
    data_sources = ["source_a"] * 1 + ["source_b"] * 8
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchWeightedRandomRefillBatchSampler(
        batch_size=4,
        data_source=dataset,
        data_config=_refill_config_from_weights({"source_a": 1.0, "source_b": 1.0}),
    )

    batches = list(sampler)

    assert len(batches) == len(sampler)
    assert all(len(batch) == 4 for batch in batches)
    used_sources = _sources_for_batch(_flatten(batches), data_sources)
    assert used_sources.count("source_a") >= 1
    assert used_sources.count("source_b") >= 8
    assert used_sources.count("source_a") > len([s for s in data_sources if s == "source_a"])


def test_weighted_random_refill_sampler_uses_more_sources_than_batch_size_over_epoch():
    data_sources = ["source_a"] * 2 + ["source_b"] * 2 + ["source_c"] * 2
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchWeightedRandomRefillBatchSampler(
        batch_size=2,
        data_source=dataset,
        data_config=_refill_config_from_weights({"source_a": 1.0, "source_b": 1.0, "source_c": 1.0}),
    )

    used_sources = _sources_for_batch(_flatten(list(sampler)), data_sources)

    assert set(used_sources) == {"source_a", "source_b", "source_c"}
    assert all(used_sources.count(source) >= 2 for source in {"source_a", "source_b", "source_c"})


def test_weighted_random_refill_sampler_skew_extra_groups_catch_combined_tail():
    weights = {
        "arxiv_mveqa_answerable": 0.185,
        "arxiv_veqa_answerable": 0.185,
        "docvqa_answerable": 0.10,
        "dude_answerable": 0.53,
    }
    data_sources = [source for source in weights for _ in range(40)]
    dataset = MockRLHFDataset(data_sources)
    config = _refill_config_from_weights(weights)
    config.batch_sampler.skew_tail_p = 0.005
    config.batch_sampler.skew_grouping = "family_prefix"
    config.batch_sampler.skew_extra_groups = {"page_heavy": ["arxiv", "docvqa"]}
    sampler = VSearchWeightedRandomRefillBatchSampler(batch_size=24, data_source=dataset, data_config=config)

    arxiv_only_high = {
        "arxiv_mveqa_answerable": 7,
        "arxiv_veqa_answerable": 7,
        "docvqa_answerable": 0,
        "dude_answerable": 10,
    }
    combined_page_heavy_high = {
        "arxiv_mveqa_answerable": 7,
        "arxiv_veqa_answerable": 7,
        "docvqa_answerable": 5,
        "dude_answerable": 5,
    }

    assert not sampler._is_skewed_batch(arxiv_only_high)
    assert sampler._is_skewed_batch(combined_page_heavy_high)


def test_weighted_random_refill_sampler_replays_old_skew_config_before_resume_boundary():
    weights = {
        "arxiv_mveqa_answerable": 0.185,
        "arxiv_veqa_answerable": 0.185,
        "docvqa_answerable": 0.10,
        "dude_answerable": 0.53,
    }
    data_sources = [source for source in weights for _ in range(40)]
    dataset = MockRLHFDataset(data_sources)

    old_config = _refill_config_from_weights(weights)
    old_config.batch_sampler.skew_tail_p = 0.005
    old_config.batch_sampler.skew_grouping = "family_prefix"
    old_sampler = VSearchWeightedRandomRefillBatchSampler(batch_size=24, data_source=dataset, data_config=old_config)
    iterator = iter(old_sampler)
    next(iterator)
    state = old_sampler.state_dict()

    new_config = _refill_config_from_weights(weights)
    new_config.batch_sampler.skew_tail_p = 0.005
    new_config.batch_sampler.skew_grouping = "family_prefix"
    new_config.batch_sampler.skew_extra_groups = {"page_heavy": ["arxiv", "docvqa"]}
    resumed_sampler = VSearchWeightedRandomRefillBatchSampler(
        batch_size=24, data_source=dataset, data_config=new_config
    )
    resumed_sampler.load_state_dict(state)

    assert resumed_sampler._legacy_replay_until_yielded
    assert "page_heavy" not in resumed_sampler._replay_skew_groups
    assert "page_heavy" in resumed_sampler._skew_groups


def test_weighted_random_refill_sampler_state_dict_resumes_remaining_batches():
    data_sources = ["source_a"] * 1 + ["source_b"] * 8
    dataset = MockRLHFDataset(data_sources)
    config = _refill_config_from_weights({"source_a": 1.0, "source_b": 1.0})
    sampler = VSearchWeightedRandomRefillBatchSampler(batch_size=4, data_source=dataset, data_config=config)

    iterator = iter(sampler)
    next(iterator)
    state = sampler.state_dict()
    remaining_batches = list(iterator)

    resumed_sampler = VSearchWeightedRandomRefillBatchSampler(batch_size=4, data_source=dataset, data_config=config)
    resumed_sampler.load_state_dict(state)

    assert list(resumed_sampler) == remaining_batches


def test_exhaustive_sampler_reads_weights_from_yaml_file(tmp_path):
    weights_file = tmp_path / "weights.yaml"
    weights_file.write_text("source_a: 1.0\nsource_b: 3.0\nsource_c: 6.0\n", encoding="utf-8")
    data_sources = ["source_a"] * 5 + ["source_b"] * 9 + ["source_c"] * 20
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchExhaustiveBatchSampler(
        batch_size=10,
        data_source=dataset,
        data_config=_config_from_weights_file(weights_file),
    )

    first_batch = next(iter(sampler))

    assert _batch_source_counts(first_batch, data_sources) == {
        "source_a": 1,
        "source_b": 3,
        "source_c": 6,
    }


def test_exhaustive_sampler_reads_nested_weights_file_and_inline_overrides(tmp_path):
    weights_file = tmp_path / "weights.yaml"
    weights_file.write_text(
        "batch_sampler:\n"
        "  weights:\n"
        "    source_a: 5.0\n"
        "    source_b: 1.0\n",
        encoding="utf-8",
    )
    data_sources = ["source_a"] * 20 + ["source_b"] * 20
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchExhaustiveBatchSampler(
        batch_size=6,
        data_source=dataset,
        data_config=_config_from_weights_file(weights_file, inline_weights={"source_b": 5.0}),
    )

    first_batch = next(iter(sampler))

    assert _batch_source_counts(first_batch, data_sources) == {
        "source_a": 3,
        "source_b": 3,
    }


def test_exhaustive_sampler_state_dict_resumes_remaining_batches():
    data_sources = ["source_a"] * 2 + ["source_b"] * 10
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchExhaustiveBatchSampler(batch_size=4, data_source=dataset, data_config=_config())

    iterator = iter(sampler)
    first_batch = next(iterator)
    state = sampler.state_dict()
    remaining_batches = list(iterator)

    resumed_sampler = VSearchExhaustiveBatchSampler(batch_size=4, data_source=dataset, data_config=_config())
    resumed_sampler.load_state_dict(state)

    assert list(resumed_sampler) == remaining_batches
    combined = first_batch + [idx for batch in remaining_batches for idx in batch]
    assert sorted(combined) == list(range(12))


def test_exhaustive_sampler_deterministic_across_instances_and_resets_epochs():
    data_sources = ["source_a"] * 12 + ["source_b"] * 12
    dataset = MockRLHFDataset(data_sources)

    sampler_a = VSearchExhaustiveBatchSampler(batch_size=6, data_source=dataset, data_config=_config())
    sampler_b = VSearchExhaustiveBatchSampler(batch_size=6, data_source=dataset, data_config=_config())

    epoch0_a = list(sampler_a)
    epoch0_b = list(sampler_b)
    assert epoch0_a == epoch0_b
    assert sorted(_flatten(epoch0_a)) == list(range(24))

    sampler_a.set_epoch(1)
    epoch1_a = list(sampler_a)
    assert sorted(_flatten(epoch1_a)) == list(range(24))
    assert epoch1_a != epoch0_a

    # Starting a new iteration after exhaustion should advance and reset positions.
    epoch2_a = list(sampler_a)
    assert sorted(_flatten(epoch2_a)) == list(range(24))
    assert epoch2_a != epoch1_a


def test_exhaustive_sampler_eventually_uses_source_with_zero_initial_quota():
    data_sources = ["source_a"] * 2 + ["source_b"] * 2 + ["source_c"] * 8
    dataset = MockRLHFDataset(data_sources)
    sampler = VSearchExhaustiveBatchSampler(batch_size=2, data_source=dataset, data_config=_config_three_sources())

    batches = list(sampler)

    assert len(sampler) == 6
    assert len(batches) == 6
    assert all(len(batch) == 2 for batch in batches)

    used_indices = [idx for batch in batches for idx in batch]
    assert sorted(used_indices) == list(range(12))

    used_sources = _sources_for_batch(used_indices, data_sources)
    assert used_sources.count("source_a") == 2
    assert used_sources.count("source_b") == 2
    assert used_sources.count("source_c") == 8


def test_exhaustive_sampler_stateful_dataloader_resume_matches_remaining_batches():
    data_sources = ["source_a"] * 2 + ["source_b"] * 2 + ["source_c"] * 8
    dataset = MockRLHFDataset(data_sources)
    config = _config_three_sources()
    sampler = VSearchExhaustiveBatchSampler(batch_size=2, data_source=dataset, data_config=config)
    loader = StatefulDataLoader(dataset, batch_sampler=sampler, num_workers=0)

    iterator = iter(loader)
    first_batch = _normalize_loader_batch(next(iterator))
    state = loader.state_dict()
    remaining_batches = [_normalize_loader_batch(batch) for batch in iterator]

    resumed_sampler = VSearchExhaustiveBatchSampler(batch_size=2, data_source=dataset, data_config=config)
    resumed_loader = StatefulDataLoader(dataset, batch_sampler=resumed_sampler, num_workers=0)
    resumed_loader.load_state_dict(state)

    assert [_normalize_loader_batch(batch) for batch in resumed_loader] == remaining_batches

    used_indices = first_batch["idx"] + [idx for batch in remaining_batches for idx in batch["idx"]]
    assert sorted(used_indices) == list(range(12))


def test_create_rl_sampler_uses_exhaustive_batch_sampler_path_with_workers_enabled():
    data_sources = ["source_a"] * 2 + ["source_b"] * 10
    dataset = MockRLHFDataset(data_sources)

    sampler = create_rl_sampler(_create_rl_sampler_config(), dataset)

    assert isinstance(sampler, VSearchExhaustiveBatchSampler)
    assert len(sampler) == 3
