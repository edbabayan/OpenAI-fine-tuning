from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    data_path = root.joinpath("data")
    output_path = root.joinpath("preprocessed_data")

    model = "gpt-4o-mini"

    system_prompt = (
        "Given the medical description report, classify it into one of these categories: "
        "[Cardiovascular / Pulmonary, Gastroenterology, Neurology, Radiology, Surgery]"
    )