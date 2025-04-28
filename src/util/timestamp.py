import time

def timestamp() -> str:
    """Gets the current timestamp as a string.

    Returns:
        str: The current timestamp in seconds since the epoch.
    """
    return str(int(time.time()))