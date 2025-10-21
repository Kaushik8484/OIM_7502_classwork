# sp500_project/items.py
import scrapy

class Sp500ProjectItem(scrapy.Item):
    # Define the fields for your item here like:
    number = scrapy.Field()       # For the rank/number column
    company = scrapy.Field()      # For the company name
    symbol = scrapy.Field()       # For the ticker symbol
    ytd_return = scrapy.Field()   # For the YTD Return column
    # pass