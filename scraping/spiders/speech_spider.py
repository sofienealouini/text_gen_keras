import scrapy
import json


class SpeechSpider(scrapy.Spider):
    name = "speech"
    custom_settings = {"FEED_EXPORT_ENCODING": 'utf-8'}

    def start_requests(self):
        with open('links.json') as data_file:
            data = json.load(data_file)
        urls = [data[i]["link"] for i in range(len(data))]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        sentences = response.xpath('//div[@class="article"]/p/text()').extract()
        sentences = [s for s in sentences if len(s) >=3]
        yield {
            'titre': response.css('h2::text').extract()[1],
            'discours': "".join(response.xpath('//div[@class="col1"]/p/text()').extract())[5:],
            'personne': sentences[0] if len(sentences)>=3 else "",
            'fonction': sentences[1] if len(sentences)>=3 else "",
            'contexte': sentences[2] if len(sentences)>=3 else ""
        }