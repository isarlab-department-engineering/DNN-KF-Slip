
class RowsParseFile(Exception):
    """Exception type for use within DatasTools classes"""

    def __init__(self, *args, **kwargs):
        default_message = 'An error with the rows of the dataset occurred!'
        if not (args or kwargs):
            args = (default_message,)

        # Call super constructor
        super().__init__(*args, **kwargs)


class RoadModel(Exception):
    """Exception type for use within RoadModels derived classes"""

    def __init__(self, *args, **kwargs):
        default_message = 'An error with the road model generation occurred!'
        if not (args or kwargs):
            args = (default_message,)

        # Call super constructor
        super().__init__(*args, **kwargs)


class MLModelException(Exception):
    """Exception type for use within MLModel derived classes"""

    def __init__(self, *args,  **kwargs):
        default_message = 'An error with MLModel class occurred!'
        if not (args or kwargs):
            args = (default_message,)

        # Call super constructor
        super().__init__(*args, **kwargs)
