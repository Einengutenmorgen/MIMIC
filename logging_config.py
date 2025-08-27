import logging
import os
import sys
import io

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Only wrap sys.stdout if it's safe (i.e. not IPython's OutStream)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler with UTF-8 encoding
f_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
f_handler.setLevel(logging.DEBUG)

# Optional: console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(f_handler)
logger.addHandler(console_handler)
