import yaml

def parse_file(file_path, section="ForecastModel"):
    """
    Parse configuration file, expecting the file written in yaml

    Parameters
    ----------
    file_path : str
        file path location

    Returns
    -------
    variables : dict
        variables that needed for idm pipeline
    """

    try:
        with open(file_path, 'r') as f:
            slt_config = yaml.safe_load(f)[section]
    except:
        raise KeyError(f"No {section} in config file")

    return slt_config
