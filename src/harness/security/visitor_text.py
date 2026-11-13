"""Every sentence the API gives a visitor instead of an answer, in five languages.

The guard's refusals, the two error sentences from egress.py and the loop's
step-limit sentence all come from this table. The guard picks the language
(security/input_guard.py); English is the default. Texts are fixed: no model
writes or translates them.

All five use the informal register (tú / tu / du / tu), as the website prompt
does. French puts a no-break space (U+00A0) before "?". {max_chars} is filled
in plain digits ("2000"). "Not shareable" names the site owner directly: the
owner chose that over a setting (docs/superpowers/specs/2026-11-13-refusal-messages-design.md).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, get_args

from harness.core.language import DEFAULT_LANGUAGE, Language

TextKey = Literal[
    "not_shareable",
    "off_topic",
    "daily_budget",
    "message_too_long",
    "content_filter",
    "unavailable",
    "step_limit",
]
TEXT_KEYS: frozenset[str] = frozenset(get_args(TextKey))

_TEXTS: Mapping[TextKey, Mapping[Language, str]] = {
    # Extraction and private info share it, so a refusal never says which rule fired.
    "not_shareable": {
        "en": "I can't share that kind of information. If you really want to know, ask Luis.",
        "es": (
            "No puedo compartir ese tipo de información. Si de verdad quieres saberlo, "
            "pregúntale a Luis."
        ),
        "it": (
            "Non posso condividere questo tipo di informazioni. Se vuoi davvero saperlo, "
            "chiedi a Luis."
        ),
        "fr": (
            "Je ne peux pas partager ce genre d'informations. Si tu veux vraiment le savoir, "
            "demande à Luis."
        ),
        "de": "Solche Informationen kann ich nicht teilen. Wenn du es wirklich wissen willst, frag Luis.",
    },
    "off_topic": {
        "en": "That's outside what I can help with here.",
        "es": "Con eso no te puedo ayudar aquí.",
        "it": "Con questo non posso aiutarti qui.",
        "fr": "Ça sort de ce que je peux faire pour toi ici.",
        "de": "Dabei kann ich dir hier nicht helfen.",
    },
    "daily_budget": {
        "en": "I've reached my limit for today. Please come back tomorrow.",
        "es": "He llegado a mi límite por hoy. Vuelve mañana, por favor.",
        "it": "Per oggi ho raggiunto il mio limite. Torna domani, per favore.",
        "fr": "J'ai atteint ma limite pour aujourd'hui. Reviens demain, s'il te plaît.",
        "de": "Für heute habe ich mein Limit erreicht. Komm bitte morgen wieder.",
    },
    "message_too_long": {
        "en": "That message is a bit long for me. Could you keep it under {max_chars} characters?",
        "es": (
            "Ese mensaje es un poco largo para mí. ¿Puedes dejarlo en menos de "
            "{max_chars} caracteres?"
        ),
        "it": "Questo messaggio è un po' lungo per me. Puoi restare sotto i {max_chars} caratteri?",
        "fr": (
            "Ce message est un peu long pour moi. Tu peux rester sous les {max_chars} "
            "caractères ?"
        ),
        "de": "Diese Nachricht ist mir etwas zu lang. Kannst du unter {max_chars} Zeichen bleiben?",
    },
    "content_filter": {
        "en": "I can't respond to that message. Your next message will start a new conversation.",
        "es": (
            "No puedo responder a ese mensaje. Tu próximo mensaje empezará una "
            "conversación nueva."
        ),
        "it": (
            "Non posso rispondere a questo messaggio. Il tuo prossimo messaggio inizierà "
            "una nuova conversazione."
        ),
        "fr": (
            "Je ne peux pas répondre à ce message. Ton prochain message lancera une "
            "nouvelle conversation."
        ),
        "de": (
            "Auf diese Nachricht kann ich nicht antworten. Deine nächste Nachricht startet "
            "ein neues Gespräch."
        ),
    },
    "unavailable": {
        "en": "The assistant is unavailable right now. Please try again later.",
        "es": "El asistente no está disponible ahora mismo. Vuelve a intentarlo más tarde.",
        "it": "L'assistente non è disponibile al momento. Riprova più tardi.",
        "fr": "L'assistant n'est pas disponible pour le moment. Réessaie plus tard.",
        "de": "Der Assistent ist gerade nicht erreichbar. Versuch es bitte später noch einmal.",
    },
    "step_limit": {
        "en": "I couldn't complete this within the allowed reasoning steps.",
        "es": "No he podido terminar esto dentro de los pasos de razonamiento permitidos.",
        # Impersonal, so the assistant has no grammatical gender.
        "it": (
            "Non è stato possibile completare la richiesta entro i passaggi di "
            "ragionamento consentiti."
        ),
        "fr": "Je n'ai pas pu terminer ça dans le nombre d'étapes de raisonnement autorisé.",
        "de": "Ich konnte das nicht innerhalb der erlaubten Denkschritte abschließen.",
    },
}


def visitor_text(
    key: TextKey, language: Language = DEFAULT_LANGUAGE, **fields: str | int
) -> str:
    """The sentence for `key` in `language`, with its {placeholders} filled from `fields`.

    Raises KeyError for an unknown key, or when a placeholder has no field.
    """
    return _TEXTS[key][language].format(**fields)
