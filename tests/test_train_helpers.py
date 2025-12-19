from src.train import build_dataset, load_positions_from_csv


def test_load_positions_small():
    path = "src/data/random_evals.csv"
    positions = load_positions_from_csv(path, max_rows=5)
    assert len(positions) == 5
    X, y = build_dataset(positions)
    assert X.shape[0] == 5
    assert X.shape[1] > 0
    assert y.shape[0] == 5
