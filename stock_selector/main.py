import datetime as dt
from loguru import logger
from pipeline import Etl


def main():
    """
    Main function
    """

    logger.remove()
    logger.add(
        rf"./log/log_{dt.datetime.now().strftime('%Y%m%d')}.log",
        backtrace=False,
        format=("{time:YYYY-MM-DD HH:mm:ss} | "
                "{level} | "
                "{module}:{function}:{line} - {message}"),
    )

    handler = Etl()
    handler.extract_stock_data()
    handler.extract_sp_500()
    """
    f_data, m_data = handler.load_stock_data()
    index_data = handler.load_index_data()
    print(f_data.shape, m_data.shape, index_data.shape)
    preprocessor = Preprocess(f_data, m_data, index_data)
    df = preprocessor.feature_engineering()
    """
    print('done')


if __name__ == "__main__":
    main()
