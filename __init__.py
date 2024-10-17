from topic_modelling_package.utils.config import get_config

# Initialize configuration first
config = get_config()

# Now set up logging using the initialized config
from topic_modelling_package.utils.logging import setup_logging

setup_logging()
