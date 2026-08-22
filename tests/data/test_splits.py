from afmc_fm.data.splits import split_patient_ids


def test_patient_split_has_no_overlap_and_is_deterministic():
    ids = [f"p{i}" for i in range(100)]
    a = split_patient_ids(ids, seed=8, train_fraction=0.7, val_fraction=0.1)
    b = split_patient_ids(ids, seed=8, train_fraction=0.7, val_fraction=0.1)
    assert a == b
    train, val, test = map(set, a)
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(ids)
