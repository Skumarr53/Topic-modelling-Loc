# centralized_nlp_package/utils/exceptions.py

class FilesNotLoadedException(Exception):
    """Exception raised when a required file cannot be loaded."""
    def __init__(self, filename: str, message: str = "Required file could not be loaded."):
        self.filename = filename
        self.message = f"{message} Filename: {filename}"
        super().__init__(self.message)


class InvalidDataException(Exception):
    """Exception raised when the data provided is invalid."""
    def __init__(self, message: str = "Invalid data provided."):
        self.message = message
        super().__init__(self.message)


class ProcessingException(Exception):
    """Exception raised during processing steps."""
    def __init__(self, step: str, original_exception: Exception):
        self.step = step
        self.original_exception = original_exception
        self.message = f"An error occurred during {step}: {str(original_exception)}"
        super().__init__(self.message)



# def load_word_list(file_path: str) -> set:
#     try:
#         with open(file_path, "r") as f:
#             words = set(line.strip().lower() for line in f if line.strip())
#         return words
#     except FileNotFoundError:
#         raise FilesNotLoadedException(filename=file_path, message="Word list file not found.")
#     except Exception as e:
#         raise ProcessingException(step="load_word_list", original_exception=e)