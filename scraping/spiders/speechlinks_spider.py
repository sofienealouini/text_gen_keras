import scrapy


class SpeechLinkSpider(scrapy.Spider):
    name = "speechlinks"

    def start_requests(self):
        base_url = "http://www.vie-publique.fr/rechercher/recherche.php?skin=cdp&replies=30&sort=-document_date_publication&b="
        urls = [(base_url + str(i)) for i in range(0, 133064, 30)]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):

        for speech in response.css('p.titre'):
            yield {
                'link': speech.css('a::attr("href")').extract_first().replace(" ", "")[:-1],
            }