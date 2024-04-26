import requests
import yaml


with open("trainer/config.yaml", "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)


def load_data() -> tuple[list, list]:
    """Function to automatic download text data from specific url in config.yaml

    Returns:
        tuple[list, list]: A tuple containing train and test lists
    """
    url = cfg["dataconfig"]["data_url"]
    text = requests.get(url).text

    # preaparing data
    text = text.split("BOOK ONE: 1805")[2].strip()
    text = text.split("\r\n\r\n")

    # filter data from symbols more then 20 and splitting
    text = list(filter(lambda x: len(x) > 20, text))
    train = text[: int(len(text) * cfg["dataconfig"]["split"])]
    test = text[int(len(text) * cfg["dataconfig"]["split"]) :]

    return train, test
