from dotenv import load_dotenv
from topic_modelling_package.utils.config import get_config

load_dotenv()
# Initialize configuration first
config = get_config()

# Now set up logging using the initialized config
from topic_modelling_package.utils.logging import setup_logging

setup_logging()
