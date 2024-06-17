from config import get_config_dict

def test_config():
    values = get_config_dict("data")['base_date']
    assert values == '2005-01-01'

if __name__ == '__main__':
    test_config()
