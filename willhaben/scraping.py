import pandas as pd
import requests
import json
from bs4 import BeautifulSoup

def get_json(page: int) -> list[dict]:
    html = requests.get(
        f"https://www.willhaben.at/iad/kaufen-und-verkaufen/marktplatz/armbanduhren-2651?isNavigation=true&srcType=vertical-browse&srcAdd=2651&sfId=e89b26f0-3f08-4d42-9bc9-588b59d135e8&rows=90&page={page}")

    parsed = BeautifulSoup(html.content, "html.parser")
    data_raw = parsed.find("script", {"id": "__NEXT_DATA__"}).text

    data = json.loads(data_raw)
    search_results = data["props"]["pageProps"]["searchResult"]["advertSummaryList"]["advertSummary"]

    return search_results

def clean_result(singe_result: dict) -> dict[str, str]:
    parsed_attributes = {
        "price": None,
        "teaser": None,
        "plz": None,
        "bundesland": None,
        "description": singe_result["description"],
        "id": singe_result["id"]
    }

    attrs = singe_result["attributes"]["attribute"]
    for attr in attrs:
        try:
            match attr["name"]:
                case "PRICE":
                    parsed_attributes["price"] = float(attr["values"][0])
                case "TEASER_ATTRIBUTE":
                    parsed_attributes["teaser"] = attr["values"][0]
                case "POSTCODE":
                    parsed_attributes["plz"] = int(attr["values"][0])
                case "STATE":
                    parsed_attributes["bundesland"] = attr["values"][0]
        except BaseException as e:
            print("error", e)
        #print(attr["name"], attr["values"])

    return parsed_attributes

def scrape(page_start: int = 1, page_end: int = 10):
    frame = {
        "price": [],
        "teaser": [],
        "plz": [],
        "bundesland": [],
        "description": []
    }

    ids = []

    for page in range(page_start, page_end+1):
        results = get_json(page)

        for raw_result in results:
            result = clean_result(raw_result)

            for k,v in result.items():
                if k == "id":
                    ids.append(v)
                    continue
                frame[k].append(v)

    return pd.DataFrame(frame, index=ids)


#print(scrape(1,1))






























# David ist toll