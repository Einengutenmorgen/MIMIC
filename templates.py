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

    
    reflect_results_template = """Untersuche die Qualität der Imitation, indem du den verwendeten Prompt und die Ergebnisse analsierst.
    Prompt: {persona}
    beste Imitation: {best_preds}
    Orginal: {best_originals}

    Schlechte Imitation: {worst_preds}
    Orginal: {worst_originals}

    durchschnittliche BLEU scores: {bleu_scores}
    durchschnittliche ROUGE scores: {rouge_scores}

    Antworte mit einer Reflection der Ergebnisse und einer verbessterten Version der Persona Beschreibung.

    Produce JSON matching this specification:

output = {{ "Reflection": string, "improved_persona": string }}
Return: <output>"""




    templates = {
        "persona_template_simple": persona_template_simple,
        "imitation_post_template_simple": imitation_post_template_simple,
        "imitation_replies_template_simple": imitation_replies_template_simple,
        "mask_opion_template": mask_opion_template,
        "reflect_results_template": reflect_results_template
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