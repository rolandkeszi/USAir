"""Anonymises data/Tweets.csv in place so that no tweet can be linked to a person.

Steps:
1. drop the author name and the tweet id columns;
2. replace @mentions of private accounts with @user (airline accounts are kept);
3. mask e-mail addresses, phone numbers, long numeric ids and booking codes;
4. mask person names found by spaCy NER (en_core_web_sm).

Run: python scripts/anonymize_tweets.py  (needs spacy and en_core_web_sm)
"""
import csv
import re
from pathlib import Path

import spacy

SRC = Path(__file__).resolve().parents[1] / "data" / "Tweets.csv"
AIRLINES = {"virginamerica", "united", "southwestair", "jetblue", "usairways", "americanair", "delta"}
FIELDS = ["airline_sentiment", "airline_sentiment_confidence", "negativereason", "negativereason_confidence",
          "airline", "retweet_count", "text", "tweet_created", "latitude", "longitude"]
NOT_PEOPLE = {
    "Virgin", "America", "United", "Southwest", "Delta", "JetBlue", "Jet", "Blue", "American", "Airways",
    "Twitter", "WiFi", "Wifi", "Passbook", "Vegas", "Austin", "Denver", "Boston", "Chicago", "Dallas", "Houston",
    "Miami", "Orlando", "Newark", "Philly", "Charlotte", "Phoenix", "Nashville", "Atlanta", "Seattle", "Portland",
    "Kudos", "Glad", "Nah", "Plane", "Bag", "Miles", "Fingers", "Tues", "Thanks", "Thank", "Cancelled", "Flight",
    "Flightled", "Gate", "Customer", "Service", "Lol", "Ugh", "Wow", "Yay", "Hey", "Hi", "Please",
}

mention = re.compile(r"@(\w+)")
email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
phone = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
long_number = re.compile(r"\b[A-Z]{0,3}\d{7,}\b")
code = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")
flight = re.compile(r"^[A-Z]{2}\d{1,4}$")
initial_name = re.compile(r"\b[A-Z][a-z]+ [A-Z]\.(?=\s|$|[,!?)])|\bDeeDee\b")  # "Mark T." style staff names NER misses
name_shape = re.compile(r"^[A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*$")


def mask_patterns(text):
    text = mention.sub(lambda m: m.group(0) if m.group(1).lower() in AIRLINES else "@user", text)
    text = email.sub("[email]", text)
    text = phone.sub("[phone]", text)
    text = long_number.sub("[number]", text)
    text = initial_name.sub("[name]", text)
    return code.sub(lambda m: m.group(0) if flight.match(m.group(0)) else "[code]", text)


def is_person(ent, text):
    if ent.label_ != "PERSON" or not name_shape.match(ent.text):
        return False
    if any(w in NOT_PEOPLE for w in re.split(r"[ '-]", ent.text)):
        return False
    before = text[max(0, ent.start_char - 1):ent.start_char]
    return before not in {"#", "@"}


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    texts = [mask_patterns(r["text"]) for r in rows]
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    masked = 0
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        t = texts[i]
        for ent in sorted((e for e in doc.ents if is_person(e, t)), key=lambda e: -e.start_char):
            t = t[:ent.start_char] + "[name]" + t[ent.end_char:]
            masked += 1
        texts[i] = t
    with open(SRC, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r, t in zip(rows, texts):
            r = dict(r, text=t)
            w.writerow({k: r[k] for k in FIELDS})
    print(f"{len(rows)} tweets anonymised, {masked} person names masked")


if __name__ == "__main__":
    main()
