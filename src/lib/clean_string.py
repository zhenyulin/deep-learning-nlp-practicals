import re


def do_nothing(input):
    return input


def remove_paren(input):
    return re.sub(r"\([^)]*\)", "", input)


def alphanumeric_period_only(input):
    return re.sub(r"[^a-z0-9\.]+", " ", input)


def regulate_punctuation(input):
    return (
        input.replace("?", ".").replace("!", ".").replace(",", " ").replace("...", " ")
    )


def dirty_remove_speaker_name(input):
    match = re.match(r"^(?:(?P<SpeakerName>[^:]{,20}):)?(?P<Content>.*)$", input)
    return match.groupdict()["Content"]


def lower_string(input):
    return input.lower()
