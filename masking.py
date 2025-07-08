from llm import call_ai
from logging_config import logger
import templates


def mask_text(text, **kwargs):
    """
    Mask all opinion-bearing words in the text
    
    :param text: The input text to be masked.
    :return: Masked text with sensitive information replaced by placeholders.
    """
    formated_text=templates.format_template("mask_opinion_template", **{"text": text})
    logger.debug(f"Masking text: {formated_text}")
    logger.info(f"masked text")
    masked_text = call_ai(formated_text)
    logger.debug(f"Masked text: {masked_text}")
    if not masked_text:
        logger.error("No masked text returned from the AI model.")
        return "" #
    return masked_text
