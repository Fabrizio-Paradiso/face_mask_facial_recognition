from bs4 import BeautifulSoup


class Soup:
    def __init__(self) -> None:
        pass

    @staticmethod
    def add_ip_address_tag(soup: BeautifulSoup, ip_address: str) -> None:
        """
        This function add ip address text in span

        Args:
            soup (BeautifulSoup): BeautifulSoup (object)
            ip_address (str): Local IPv4 address
        Return:
            None
        """
        new_tag = soup.new_tag("a", id="ip-address")
        new_tag.string = f"{ip_address}"
        container_tag = soup.find("span", {"id": "sub-title-span"})
        container_tag.append(new_tag)

    @classmethod
    def insert_ip_address(
        cls, ip_address: str, path_file: str = f"templates/index.html"
    ) -> None:
        """
        This function insert IPv4 address in HTML

        Args:
            ip_address (str): Local IPv4 address
            path_file (str): Path where HTML is located
        Return:
            None
        """
        contents = open(path_file, "r").read()
        soup = BeautifulSoup(contents, "lxml")

        if soup.find("a", {"id": "ip-address"}).get_text() is not ip_address:
            soup.find("a", {"id": "ip-address"}).extract()
            cls.add_ip_address_tag(soup, ip_address)

        if soup.find("a", {"id": "ip-address"}) is None:
            cls.add_ip_address_tag(soup, ip_address)

        savechanges = soup.prettify("utf-8")
        with open(path_file, "wb") as file:
            file.write(savechanges)
