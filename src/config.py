from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    data_path = root.joinpath("data")
