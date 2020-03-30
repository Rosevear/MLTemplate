def featurize_data(data):
    """
    Processes the provided pandas dataframe object by:
    
    Parsing the dates into separate columns (month and day) and deleting the original READ_DTM and METER_ID columns
    """

    try:
        data['DAY'] = pd.DatetimeIndex(data['READ_DTM']).day
        data['MONTH'] = pd.DatetimeIndex(data['READ_DTM']).month
        del data['READ_DTM']
        del data['METER_ID']
        return data
    except Exception as e:
        logger.info(
            'There was a problem processing the data set: {}'.format(str(e)))
