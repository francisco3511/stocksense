from config import config


def test_config():
    values = config.scraping.base_date
    assert values == "2005-01-01"


if __name__ == "__main__":
    test_config()
