"""Official publisher RSS feeds (headlines and blurbs only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Feed:
    outlet: str
    section: str
    url: str


# Section feeds only. Full article HTML is never fetched.
FEEDS: tuple[Feed, ...] = (
    # CNN
    Feed("CNN", "top_stories", "http://rss.cnn.com/rss/cnn_topstories.rss"),
    Feed("CNN", "world", "http://rss.cnn.com/rss/cnn_world.rss"),
    Feed("CNN", "us", "http://rss.cnn.com/rss/cnn_us.rss"),
    Feed("CNN", "politics", "http://rss.cnn.com/rss/cnn_allpolitics.rss"),
    Feed("CNN", "technology", "http://rss.cnn.com/rss/cnn_tech.rss"),
    Feed("CNN", "health", "http://rss.cnn.com/rss/cnn_health.rss"),
    # The New York Times
    Feed("New York Times", "home", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
    Feed("New York Times", "world", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    Feed("New York Times", "us", "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"),
    Feed("New York Times", "business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    Feed(
        "New York Times",
        "technology",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ),
    # The Washington Times
    Feed("Washington Times", "news", "https://www.washingtontimes.com/rss/headlines/news/"),
    Feed("Washington Times", "opinion", "https://www.washingtontimes.com/rss/headlines/opinion/"),
    Feed("Washington Times", "sports", "https://www.washingtontimes.com/rss/headlines/sports/"),
    Feed("Washington Times", "culture", "https://www.washingtontimes.com/rss/headlines/culture/"),
    # Hindustan Times
    Feed(
        "Hindustan Times",
        "india",
        "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    ),
    Feed(
        "Hindustan Times",
        "world",
        "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml",
    ),
    Feed(
        "Hindustan Times",
        "latest",
        "https://www.hindustantimes.com/feeds/rss/latest/rssfeed.xml",
    ),
    Feed(
        "Hindustan Times",
        "cities",
        "https://www.hindustantimes.com/feeds/rss/cities/rssfeed.xml",
    ),
    # The Times of India
    Feed(
        "Times of India",
        "top_stories",
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    ),
    Feed(
        "Times of India",
        "india",
        "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    ),
    Feed(
        "Times of India",
        "world",
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
    ),
    Feed(
        "Times of India",
        "business",
        "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
    ),
    # The Indian Express
    Feed("Indian Express", "india", "https://indianexpress.com/section/india/feed/"),
    Feed("Indian Express", "world", "https://indianexpress.com/section/world/feed/"),
    Feed("Indian Express", "cities", "https://indianexpress.com/section/cities/feed/"),
    Feed("Indian Express", "technology", "https://indianexpress.com/section/technology/feed/"),
    # Wall Street Journal (public Dow Jones feeds)
    Feed(
        "Wall Street Journal",
        "world",
        "https://feeds.content.dowjones.io/public/rss/RSSWorldNews",
    ),
    Feed(
        "Wall Street Journal",
        "opinion",
        "https://feeds.content.dowjones.io/public/rss/RSSOpinion",
    ),
    Feed(
        "Wall Street Journal",
        "technology",
        "https://feeds.content.dowjones.io/public/rss/RSSWSJD",
    ),
    Feed(
        "Wall Street Journal",
        "economy",
        "https://feeds.content.dowjones.io/public/rss/socialeconomyfeed",
    ),
    Feed(
        "Wall Street Journal",
        "markets",
        "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    ),
    # BBC
    Feed("BBC", "top", "https://feeds.bbci.co.uk/news/rss.xml"),
    Feed("BBC", "world", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    Feed("BBC", "technology", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    Feed("BBC", "business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    # The Guardian
    Feed("The Guardian", "world", "https://www.theguardian.com/world/rss"),
    Feed("The Guardian", "uk", "https://www.theguardian.com/uk/rss"),
    Feed("The Guardian", "technology", "https://www.theguardian.com/technology/rss"),
    Feed("The Guardian", "business", "https://www.theguardian.com/uk/business/rss"),
    # The Hindu
    Feed("The Hindu", "national", "https://www.thehindu.com/news/national/feeder/default.rss"),
    Feed(
        "The Hindu",
        "international",
        "https://www.thehindu.com/news/international/feeder/default.rss",
    ),
    Feed("The Hindu", "home", "https://www.thehindu.com/feeder/default.rss"),
)
