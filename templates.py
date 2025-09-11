def select_template(template_name):

    persona_template_simple= """
"Create an accurate user profile based on the user's Twitter posts:

{historie}

Use the following format:
-[FEATURE_NAME]:: <<Description>>
-[FEATURE_NAME]:: <<Description>>
..."""

    imitation_post_template_simple  = """You are mimicking the writing style and voice of a specific Twitter user. Based on their persona description, fill in the [MASKED] words in the tweet below to match their authentic voice, vocabulary, and typical expressions.

Persona: {persona}

Original Tweet: {tweet}

Instructions:
- Replace each [MASKED] token with words that authentically reflect this user's writing style
- Maintain the original tweet's structure and meaning
- Use vocabulary and phrases this person would naturally use
- Keep the tone consistent with their typical posts

Completed Tweet:"""

    imitation_replies_template_simple = """You are responding to a tweet as a specific Twitter user. Based on their persona description, craft a reply that authentically matches their voice, opinions, and typical engagement style.

Persona: {persona}

Tweet to reply to: {tweet}

Instructions:
- Write a reply that this user would naturally post
- Match their typical tone, vocabulary, and communication style
- Consider their likely opinions and reactions to this type of content
- Keep the response length appropriate for Twitter
- Ensure the reply feels genuine to their established persona

Reply:"""

    # Improved Template
    mask_opinion_template = """
CRITICAL REQUIREMENT: You MUST replace at least 1 word with [MASKED]!

Task: Mask strongly opinionated and evaluative words in the following text with [MASKED].

IMPORTANT RULES:
1. AT LEAST 1 word MUST be masked - this is mandatory!
2. Leave proper names unchanged (Biden, Trump, California, etc.)
3. Leave neutral nouns unchanged (voters, election, people, etc.)
4. Leave numbers and URLs unchanged
5. Mask primarily: strongly evaluative adjectives, derogatory terms, emotional expressions
6. If no obvious opinion words are present, mask the most subjective available word
7. Return ONLY the masked text, no explanations
8. NEVER respond with "I cannot" or "there are none"

Examples:
- "Biden is a great president" → "Biden is a [MASKED] president"  
- "Trump is terrible" → "Trump is [MASKED]"
- "California needs help" → "California needs [MASKED]" (if necessary)

Text: {text}


Masked text (with at least 1 [MASKED]):"""


    reflect_results_template = """Examine the quality of the imitation by analyzing the used prompt and the results.
    Prompt: {persona}
    Best imitation: {best_preds}
    Original: {best_originals}

    Poor imitation: {worst_preds}
    Original: {worst_originals}

    Metric Scores:
{metric_scores}

    Respond with a reflection of the results and an improved version of the persona description.

    Produce JSON matching this specification:

output = {{ "Reflection": string, "improved_persona": string }}
Return: <output>"""




    llm_evaluation_template = """
    Please evaluate the coherence and consistency of the following 'prediction' text based on the 'reference' text.
    Coherence refers to how well the prediction flows and makes sense on its own.
    Consistency refers to how well the prediction aligns with the facts and tone of the reference.

    Reference: "{reference}"
    Prediction: "{prediction}"

    Provide a score from 1 to 5 for both coherence and consistency and a brief justification.
    Format your response as:
    Coherence: [score]
    Consistency: [score]
    Justification: [text]
    """

    imitation_post_template_no_persona = """Fill in the [MASKED] words in the tweet below to create a natural and coherent message:

Original Tweet: {tweet}

Instructions:
- Replace each [MASKED] token with words that make the sentence flow naturally
- Maintain the original tweet's structure and meaning
- Use clear, appropriate language for social media
- Ensure the completed tweet reads as a normal social media post

Completed Tweet:"""

    imitation_replies_template_no_persona = """Write a natural reply to the following tweet:

Tweet to reply to: {tweet}

Instructions:
- Write a relevant and appropriate reply to this content
- Keep the response length suitable for social media
- Make it conversational and engaging
- Ensure the reply feels authentic and natural

Reply:"""

    # GENERIC-PERSONA Templates (minimal, universal persona)
    imitation_post_template_generic = """You are a typical social media user who communicates naturally online. Fill in the [MASKED] words in the tweet below to match a natural social media communication style.

Persona: You are an active social media user who writes posts and replies in a conversational, authentic way. You engage with various topics and express yourself clearly and naturally online.

Original Tweet: {tweet}

Instructions:
- Replace each [MASKED] token with words that reflect natural social media communication
- Maintain the original tweet's structure and meaning
- Use vocabulary and tone typical for social media posts
- Keep the style conversational and authentic

Completed Tweet:"""

    imitation_replies_template_generic = """You are a typical social media user who communicates naturally online. Write a reply that matches natural social media engagement patterns.

Persona: You are an active social media user who writes posts and replies in a conversational, authentic way. You engage with various topics and express yourself clearly and naturally online.

Tweet to reply to: {tweet}

Instructions:
- Write a reply that reflects natural social media communication
- Match typical social media tone and engagement style
- Consider how people generally respond to this type of content
- Keep the response length appropriate for social media
- Ensure the reply feels genuine and conversational

Reply:"""
    imitation_post_template_no_persona = """You are completing a social media post. Fill in the [MASKED] words in the tweet below to complete it naturally.

    Original Tweet: {tweet}

    Instructions:
    - Replace each [MASKED] token with appropriate words
    - Maintain the original tweet's structure and meaning
    - Write naturally and authentically
    - Keep the tone consistent

    Completed Tweet:"""

    imitation_replies_template_no_persona = """You are responding to a social media post. Write a natural reply to the tweet below.

Tweet to reply to: {tweet}

Instructions:
- Write a natural reply that someone would post
- Keep the response length appropriate for social media
- Write authentically and conversationally

Reply:"""

    templates = {
        "persona_template_simple": persona_template_simple,
        "imitation_post_template_simple": imitation_post_template_simple,
        "imitation_replies_template_simple": imitation_replies_template_simple,
        "mask_opinion_template": mask_opinion_template,
        "reflect_results_template": reflect_results_template,
        "llm_evaluation_template": llm_evaluation_template,
        "imitation_post_template_no_persona": imitation_post_template_no_persona,
        "imitation_replies_template_no_persona": imitation_replies_template_no_persona,
        "imitation_post_template_generic": imitation_post_template_generic,
        "imitation_replies_template_generic": imitation_replies_template_generic,
        "imitation_post_template_no_persona": imitation_post_template_no_persona,
        "imitation_replies_template_no_persona": imitation_replies_template_no_persona
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