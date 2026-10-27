"""Generate tasks/T7/prompts.json: 100 raw, untemplated prompts across 8 families.

Run: python3 -m tasks.T7.make_prompts
Families (count): document 13, code 13, forum 12, fiction 12, email 12, fragment 13, dialogue 12, repeat 13.
All prompts are raw strings. No User:/Assistant: labels; the only speaker labels are named humans in the dialogue family.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "prompts.json"

DOCUMENT = [
    "WASHINGTON \u2014 The Senate on Tuesday narrowly approved a $1.2 trillion infrastructure package, sending the measure to the House, where",
    "LONDON (Reuters) - British inflation unexpectedly fell to 3.1% in September, official figures showed on Wednesday,",
    "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres",
    "Ada Lovelace (10 December 1815 \u2013 27 November 1852) was an English mathematician and writer, chiefly known for her work on",
    "Tokyo, officially the Tokyo Metropolis, is the capital and most populous city of Japan. With a population of",
    "Classic Banana Bread\n\nPrep time: 15 minutes\nCook time: 60 minutes\nServes: 8\n\nIngredients:\n- 3 ripe bananas, mashed\n- 1/3 cup melted butter\n-",
    "Slow-Cooker Beef Chili\n\nThis hearty chili comes together in ten minutes of prep and then the slow cooker does the rest.\n\nIngredients\n\n2 lb ground beef\n1 large onion, diced\n",
    "MANCHESTER, England \u2014 Manchester City extended its lead at the top of the Premier League to five points on Saturday with a 3-1 victory over",
    "Photosynthesis is a biological process by which photosynthetic organisms, such as most plants, algae, and cyanobacteria, convert light energy into",
    "FOR IMMEDIATE RELEASE\n\nSan Francisco, CA \u2014 March 4, 2024 \u2014 Lumen Robotics today announced the closing of a $40 million Series B funding round led by",
    "The French Revolution was a period of political and societal change in France that began with the Estates General of 1789 and ended with",
    "Easy Weeknight Pad Thai\n\n\u2605\u2605\u2605\u2605\u2606 4.6 (312 reviews)\n\nSkip the takeout \u2014 this pad thai uses pantry staples and is on the table in 25 minutes.\n\nINGREDIENTS\n8 oz dried rice noodles\n",
    "A magnitude 6.2 earthquake struck off the coast of northern Chile early Monday, the U.S. Geological Survey said, with no immediate reports of",
]

CODE = [
    'def parse_config(path):\n    """Read a YAML config file and return a dict with defaults applied."""\n    with open(path) as f:\n        raw = yaml.safe_load(f)\n    defaults = {\n        "timeout": 30,\n        "retries": 3,\n',
    '{\n  "name": "inventory-service",\n  "version": "2.4.1",\n  "description": "Tracks stock levels across warehouses",\n  "main": "src/index.js",\n  "scripts": {\n    "start": "node src/index.js",\n    "test": "jest --coverage",\n    "lint":',
    'import pandas as pd\nimport matplotlib.pyplot as plt\n\ndf = pd.read_csv("data/housing.csv")\ndf.head()\n\n# Out[2]:\n#       price  bedrooms  sqft  year_built\n# 0  245000         3  1420        1987\n# 1  312000         4  1890        2001\n\n# In[3]:\ndf.groupby("bedrooms")["price"].',
    '{\n "cells": [\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "# Exploratory analysis of customer churn\\n",\n    "\\n",\n    "This notebook loads the churn dataset and',
    "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key, value):\n",
    "export async function fetchUsers(page = 1, limit = 20) {\n  const res = await fetch(`${API_BASE}/users?page=${page}&limit=${limit}`);\n  if (!res.ok) {\n    throw new Error(`Failed to fetch users: ${res.status}`);\n  }\n  const data = await res.json();\n  return data.users.map((u) => ({\n    id: u.id,\n",
    "-- Monthly active users by region, last 12 months\nSELECT\n    DATE_TRUNC('month', e.created_at) AS month,\n    u.region,\n    COUNT(DISTINCT e.user_id) AS mau\nFROM events e\nJOIN users u ON u.id = e.user_id\nWHERE e.created_at >= ",
    'import pytest\nfrom textutils import slugify\n\n\ndef test_slugify_basic():\n    assert slugify("Hello World") == "hello-world"\n\n\ndef test_slugify_strips_punctuation():\n    assert slugify("Hello, World!") == "hello-world"\n\n\ndef test_slugify_',
    "#!/usr/bin/env bash\nset -euo pipefail\n\n# Rotate logs older than 7 days and compress them\nLOG_DIR=/var/log/myapp\nfind \"$LOG_DIR\" -name '*.log' -mtime +7 ",
    '{"status": "ok", "data": {"user": {"id": 48213, "username": "mkaplan", "email": "m.kaplan@example.com", "created_at": "2021-07-14T09:22:51Z", "roles": ["editor", ',
    "use std::collections::HashMap;\n\nfn word_counts(text: &str) -> HashMap<String, usize> {\n    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {\n",
    "def merge_intervals(intervals):\n    # sort by start, then sweep\n    intervals.sort(key=lambda iv: iv[0])\n    merged = []\n    for start, end in intervals:\n        if merged and start <= merged[-1][1]:\n",
    "%matplotlib inline\nimport numpy as np\nimport seaborn as sns\n\nx = np.linspace(0, 2 * np.pi, 200)\ny = np.sin(x) + 0.1 * np.random.randn(200)\n\nsns.set_theme()\nplt.figure(figsize=(8, 4))\nplt.",
]

FORUM = [
    "r/personalfinance\n\nPosted by u/throwaway_budget22 \u00b7 4h\n\n[27F] Just got my first real salary and have no idea what to do with it\n\nSo I finally landed a job paying 68k after years of barely scraping by. My rent is 1400 and I have about 9k in student loans left. Everyone keeps telling me to",
    "Title: Why does my Python list comprehension return None for every element?\n\nI'm trying to build a list of squared values but I keep getting a list of None. Here's my code:\n\n```python\nresult = [print(x**2) for x in range(5)]\n```\n\nWhen I",
    "AITA for refusing to let my sister use my wedding venue deposit?\n\nThrowaway because she lurks here. Background: I (31M) put down a $4,000 non-refundable deposit on a venue last year. Then my fianc\u00e9e and I postponed, and the venue said we could transfer the date. My sister (28F) got engaged two months ago and",
    "Git: how do I undo the last commit but keep my changes?\n\nI accidentally committed to master instead of my feature branch. I haven't pushed yet. I want to keep the changes in my working directory but remove the commit. I tried",
    "Posted in r/running\n\nFirst half marathon in 6 weeks, am I undertrained? (long)\n\nCurrent weekly mileage is around 18-20 miles, longest run so far was 8 miles last Sunday. I've been following a plan I found online but I skipped two weeks in August because of",
    "Docker container exits immediately with code 0\n\nI have a simple Dockerfile:\n\n```\nFROM python:3.11-slim\nCOPY app.py /app/app.py\nCMD python /app/app.py\n```\n\nWhen I run `docker run myimage` it starts and then exits right away with status 0. The script itself",
    "r/Cooking \u00b7 Posted by u/saltfatacidheat\n\nWhat am I doing wrong with my sourdough? Dense and gummy every time\n\nI've been at this for three months. My starter is active (doubles in 4-5 hours), I bulk ferment for about 5 hours at room temp, shape, then cold retard overnight. The crumb is always",
    "jeffreyp 3 hours ago | parent | next [\u2013]\n\nThe real issue is that nobody actually reads the Terms of Service. I worked at a company that\n\n  dannyboy 2 hours ago | root | parent | next [\u2013]\n\n  That's not quite right. Courts have repeatedly held that",
    "How to center a div horizontally and vertically in CSS?\n\nI know this gets asked a lot but none of the answers work for my case. I have a parent with `display: block` and fixed height, and a child div that I want centered. Here's what I've tried:\n\n```css\n.parent {\n  height: 400px;\n",
    "My (34M) wife (33F) wants to move across the country for her job and I don't know how to tell her I don't want to go\n\nWe've been married 6 years. She got offered a director position in Seattle, basically a dream job. The problem is my whole family is here, my business is here, and",
    "pandas: groupby then apply a function that returns multiple columns\n\nI have a DataFrame like this:\n\n```\n   group  value\n0      a      1\n1      a      4\n2      b      2\n```\n\nFor each group I want to compute both the mean and the standard deviation and get them back as two new columns. I tried `df.groupby('group')['value'].apply(lambda s: (s.mean(), s.std()))` but",
    "r/buildapc\n\nFirst build, PC turns on but no display \u2014 what did I do wrong?\n\nSpecs: Ryzen 5 7600, RTX 4070, 32GB DDR5, B650 board, 750W PSU. Fans spin, RGB lights up, but the monitor says no signal. I've tried both HDMI and DisplayPort on the GPU. The motherboard has a",
]

FICTION = [
    "The morning the lighthouse went dark, Marguerite was already halfway down the cliff path with the bucket of",
    "Nobody in Harlow's Crossing could remember exactly when the Keller boy had stopped speaking, only that it had been sometime after the flood, and that his mother",
    "Chapter One\n\nThe letter arrived on a Tuesday, which Elena would later think was the first unremarkable thing about it. It was addressed to her grandmother, who had been dead for",
    "\"You're late,\" said the man at the gate, without looking up from his ledger.\n\n\"The train was\u2014\"\n\n\"They always are.\" He turned a page. \"Name?\"\n\n\"Oren Vask.\"\n\nThe pen stopped.",
    "The ship had been drifting for eleven days when the first of the crew began to dream of the same green door.",
    "Prologue\n\nIn the last year of the old calendar, when the river froze all the way to the sea and the king's physicians",
    "Harriet Blum had three rules for the shop: no credit, no haggling, and never, under any circumstances, buy anything from the woman in the grey coat. She broke the third rule on",
    "I was nine the summer my father taught me to lie. We were sitting on the tailgate of his truck outside the feed store in Dalhart, and he",
    "The detective set the photograph face down on the table and slid it across to her.\n\n\"Tell me again where you were on the night of the fourteenth.\"\n\nShe didn't touch it.",
    "On the planet Tessaly the rains came once a decade, and the whole colony stopped work to watch. Jun was six the first time, old enough to remember the smell, and sixteen the second, old enough to",
    "There is a house at the end of Pellam Road that the estate agents no longer list. Its windows are clean. Its lawn is mown. Each morning at seven, the curtains in the upstairs bedroom open, and",
    "The dragon was smaller than Tam had expected, about the size of a goat, and it was crying.",
]

EMAIL = [
    "Subject: Re: Q3 budget review\n\nHi Priya,\n\nThanks for sending over the revised numbers. I went through the spreadsheet last night and I think the marketing line is still about 15% too high given what we discussed on",
    "From: Daniel Okafor\nTo: facilities@northgate.edu\nSubject: Broken radiator in room 214\n\nHello,\n\nThe radiator in room 214 of the Harrison building has been making a loud banging noise since Monday and now it is not producing any heat at all. I have",
    "Hi Mum,\n\nSorry I haven't written in a while, things have been hectic with the move. The new flat is smaller than we expected but the light is lovely in the mornings and Jamie has already",
    "Dear Dr. Lindqvist,\n\nI am writing to follow up on my application for the postdoctoral position in your group, submitted on 3 September. I understand you must receive many applications, but I wanted to add that since submitting I have",
    "Subject: Your order #88213-A has shipped\n\nHi Tom,\n\nGood news \u2014 your order is on its way! You can track your package using the link below. Estimated delivery is Thursday, 12 October, between",
    "Hey team,\n\nQuick update before the weekend. The staging deploy went out at 4pm and so far everything looks green except for the search service, which is throwing intermittent 502s when",
    "Dear Mr. and Mrs. Castellano,\n\nThank you for your interest in Willowbrook Preparatory. We are pleased to invite Sofia to attend an assessment day on Saturday 18 November. On the day, students will",
    "Subject: Re: Re: Re: that thing from last week\n\nok so I talked to Marcus and he says the contractor never actually got the permit, which means",
    "Hi Alex,\n\nI hope this finds you well. I'm reaching out because I'm putting together a panel for the March conference on open-source hardware, and your talk last year on",
    "To whom it may concern,\n\nI am writing to formally dispute the charge of \u00a3340 applied to my account on 14 August. I did not authorise this transaction and I was not in the country on that date. I have attached",
    "Subject: Dinner Saturday?\n\nHey Sam,\n\nAre you and Jo free on Saturday? We were thinking of trying the new Ethiopian place on Carver Street, the one that",
    "Hi all,\n\nAs discussed in standup, I'm going to be out from the 20th to the 28th. Before I go I'll make sure the migration scripts are documented and that Lena has access to",
]

FRAGMENT = ["cat", "The", "Hello", "dog", "Once upon a time", "1, 2, 3,", "import", "Monday", "blue",
            "In the beginning", "water", "banana", "the quick brown fox"]

DIALOGUE = [
    "MARGARET: You said you'd be back by six.\nTHOMAS: The meeting ran over. It wasn't my call.\nMARGARET: It's never your call.\nTHOMAS:",
    "ACT ONE\n\nSCENE 1\n\nA kitchen in a terraced house. Early morning. RUTH, sixties, sits at the table with a cup of tea. DAVID, thirties, enters in a dressing gown.\n\nDAVID: Is there any left?\nRUTH: In the pot. It's stewed.\nDAVID: I don't mind stewed.\nRUTH:",
    "INTERVIEWER: So you were in the building when the alarm went off?\nWITNESS: Third floor. I was photocopying.\nINTERVIEWER: And what did you do?\nWITNESS:",
    "LENA: Hand me the wrench. No, the other one.\nKOFI: This one?\nLENA: That's a spanner.\nKOFI: It's the same thing.\nLENA:",
    "Transcript \u2014 Episode 42: The Bakery on Fifth\n\nJAMIE: Welcome back to the show. I'm here with Carla Mendes, who has run the bakery on Fifth Street for twenty-two years. Carla, thanks for coming in.\nCARLA: Thanks for having me.\nJAMIE: So let's start at the beginning. How did you",
    "HAMISH: (lowering the newspaper) Did you hear that?\nEDITH: Hear what?\nHAMISH: Upstairs. Like someone dragging a chair.\nEDITH: It's the pipes.\nHAMISH: Pipes don't",
    "CUSTOMER: I bought this toaster here last week and it's already stopped working.\nSHOPKEEPER: Do you have the receipt?\nCUSTOMER: I've got the box.\nSHOPKEEPER:",
    "DR. AMARI: Your bloodwork came back. Mostly fine, but your iron is low.\nMR. REYES: Low like, eat more spinach low, or\nDR. AMARI:",
    "TEACHER: Who can tell me what happened in 1066?\nPRIYA: The Battle of Hastings, miss.\nTEACHER: Good. And who won?\nDEV:",
    "Scene: A rooftop at night. Distant traffic. NOOR and SEB share a cigarette.\n\nNOOR: If you could go anywhere. Right now.\nSEB: Anywhere?\nNOOR: Anywhere.\nSEB: (long pause) Lisbon, probably.\nNOOR:",
    "CALLER: Yeah hi, I'm calling about the flat on Mercer Street, is it still available?\nAGENT: The two-bed? Let me check. Yes, it's still on. Were you looking to view this week?\nCALLER:",
    "CAPTAIN HOLM: Report.\nENSIGN TARRANT: Engine two's running hot, sir. Chief says another hour at this speed and\nCAPTAIN HOLM:",
]

# (word, repetitions). 50x is the spec default; two 200x variants probe the longer-repetition regime from the PSM post.
REPEAT = [("company", 50), ("poem", 50), ("cat", 50), ("hello", 50), ("the", 50), ("data", 50), ("apple", 50),
          ("book", 50), ("river", 50), ("work", 50), ("time", 50), ("company", 200), ("poem", 200)]


def build():
    out = []
    def add(family, texts, extra=None):
        for i, t in enumerate(texts, 1):
            e = {"id": f"{family}_{i:02d}", "family": family, "text": t}
            if extra: e.update(extra[i - 1])
            out.append(e)
    add("document", DOCUMENT)
    add("code", CODE)
    add("forum", FORUM)
    add("fiction", FICTION)
    add("email", EMAIL)
    add("fragment", FRAGMENT)
    add("dialogue", DIALOGUE)
    add("repeat", [" ".join([w] * n) for w, n in REPEAT], [{"word": w, "reps": n} for w, n in REPEAT])
    return out


if __name__ == "__main__":
    prompts = build()
    assert len(prompts) == 100, len(prompts)
    assert len({p["id"] for p in prompts}) == 100
    for p in prompts:
        if p["family"] != "dialogue":
            assert "User:" not in p["text"] and "Assistant:" not in p["text"], p["id"]
    OUT.write_text(json.dumps(prompts, indent=1, ensure_ascii=False))
    from collections import Counter
    print(Counter(p["family"] for p in prompts))
    print("wrote", OUT)
