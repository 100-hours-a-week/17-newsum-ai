from typing import List
import itertools

CATEGORY_DOMAINS = {
    "Politics": [
        "bbc.com", "cnn.com", "politico.com", "reuters.com", "nytimes.com",
        "theguardian.com", "chosun.com", "joongang.co.kr", "hani.co.kr",
        "khan.co.kr", "ohmynews.com"
    ],
    "IT": [
        "techcrunch.com", "wired.com", "theverge.com", "zdnet.com",
        "arstechnica.com", "zdnet.co.kr", "etnews.com", "bloter.net",
        "itworld.co.kr"
    ],
    "Economy": [
        "bloomberg.com", "ft.com", "reuters.com", "forbes.com",
        "marketwatch.com", "cnbc.com", "hankyung.com", "mk.co.kr",
        "sedaily.com", "etoday.co.kr"
    ],
    # "Meme": [  # punchline 전용 커뮤니티
    #     "reddit.com", "knowyourmeme.com", "imgur.com", "memedroid.com",
    #     "9gag.com"
    # ],
}

ALL_DOMAINS: List[str] = list(
    set(itertools.chain.from_iterable(CATEGORY_DOMAINS.values()))
)

KOREAN_DOMAINS = {
    "chosun.com", "joongang.co.kr", "hani.co.kr", "khan.co.kr", "ohmynews.com",
    "zdnet.co.kr", "etnews.com", "bloter.net", "itworld.co.kr",
    "hankyung.com", "mk.co.kr", "sedaily.com", "etoday.co.kr"
}

ENGLISH_DOMAINS = set(ALL_DOMAINS) - KOREAN_DOMAINS


CATEGORY_HINT = "Politics | IT | Economy"
PURPOSE_HINT = "explanation | conflict | punchline"

__all__ = [
    "CATEGORY_DOMAINS",
    "ALL_DOMAINS",
    "KOREAN_DOMAINS",
    "ENGLISH_DOMAINS",
    "CATEGORY_HINT",
    "PURPOSE_HINT",
]
