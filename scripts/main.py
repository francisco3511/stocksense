import datetime as dt
from loguru import logger
from pipeline import Etl, Stock, Preprocess


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

    handler = Etl(False)
    
    #print('Updating stock data')
    #handler.extract_stock_data()
    #handler.extract_sp_500()
    
    data = handler.load_stock_data()
    preprocessor = Preprocess(data)
    df = preprocessor.feature_engineering()
    
    print('done')
    


def test_individual():
    
    stock = Stock('HRB')

    # update stock data
    stock.retrieve_insider_data(update=False)

if __name__ == "__main__":
    main()
