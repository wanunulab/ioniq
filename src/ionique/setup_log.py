import json
import functools
import logging
import numpy as np
import datetime


class JSONLogger:
    def __init__(self):
        """
        initialize logging

        """
        timestamp_init = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{timestamp_init}_log.json"
        self._clear_log_file()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _clear_log_file(self):
        """
        Clear the log file each time the kernel restarts
        When the JSONLogger init it dumps an empty lit to the file
        """
        with open(self.filename, "w") as file:
            json.dump([], file)

    def _log_to_json(self, entry):
        """
        Append the new log entry to the JSON file

        """
        with open(self.filename, "r+") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
            data.append(entry)
            file.seek(0)
            json.dump(data, file, indent=4)

    def log(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            """

            Avoid logging current and voltage arrays
            replacing numpy arrays with a placeholder

            """

            args_repr = []
            for a in args:
                if isinstance(a, np.ndarray):  # skip current numpy arrays
                    args_repr.append("<numpy array>")
                elif isinstance(a, list):  #  voltage list
                    args_repr.append("<numpy array>")
                else:
                    try:
                        args_repr.append(repr(a))
                    except Exception:
                        args_repr.append("<error in repr>")
            # Some arguments are objects that are not serializable.
            # Convert them to a string, otherwise
            kwargs_repr = []
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    kwargs_repr.append(f"{k}=<numpy array>")
                else:
                    try:
                        kwargs_repr.append(f"{k}={repr(v)}")
                    except Exception:
                        kwargs_repr.append(f"{k}=<error in repr. arg=object>")

            signature = ", ".join(args_repr + kwargs_repr)

            # log class and method name if it's a method within a class
            class_name = args[0].__class__.__name__ if args else ""
            method_name = func.__name__

            # log entries
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "class": class_name,
                "method": method_name,
                "arguments": signature
            }

            # dump log the entry to JSON
            self._log_to_json(entry)

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.logger.exception(f"Exception raised in {method_name}. Exception: {str(e)}")
                raise e

        return wrapper


json_logger = JSONLogger()
