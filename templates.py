def select_template(template_name):

    persona_template_simple= """
"Erstelle ein genaues Nutzerprofil basierend auf Twitter posts des Nutzers:

{historie}

Verwende dabei folgende Form:
-[FEATURE_NAME]:: <<Beschreibung>>
-[FEATURE_NAME]:: <<Beschreibung>>
..."""

    imitation_post_template_simple = """Basierend auf dem bereitgestellten Perona eines Twitterusers, vervollständige den folgenden Tweet:
Perosna: {persona}

    Tweet: {tweet}"""

    imitation_replies_template_simple = """Basierend auf dem bereitgestellten Perona eines Twitterusers, antworte auf den folgenden Tweet:
Perosna: {persona}
Tweet: {tweet}"""

    mask_opion_template= """Ersetze alle meinungstragenden wörter in dem folgenden Text durch [MASKED]:
    Text: {text}
    Gebe nur den maskierten Text zurück, ohne zusätzliche Erklärungen oder Formatierungen."""

    




    templates = {
        "persona_template_simple": persona_template_simple,
        "imitation_post_template_simple": imitation_post_template_simple,
        "imitation_replies_template_simple": imitation_replies_template_simple,
        "mask_opion_template": mask_opion_template
    }
    
    if template_name in templates:
        return templates[template_name]
    else:
        raise ValueError(f"Template '{template_name}' not found.")

def format_template(template_name, **kwargs):
    """
    Format a template with the provided keyword arguments.
    
    :param template_name: Name of the template to format.
    :param kwargs: Keyword arguments to fill in the template.
    :return: Formatted string based on the template.
    """
    template = select_template(template_name)
    return template.format(**kwargs)