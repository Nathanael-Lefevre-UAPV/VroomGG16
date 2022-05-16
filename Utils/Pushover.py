import http.client
import urllib
import os


def pushover(message):
    if "PUSHOVER_API_KEY" in os.environ and "PUSHOVER_USER" in os.environ:
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
                     urllib.parse.urlencode({
                         "token": os.environ["PUSHOVER_API_KEY"],
                         "user": os.environ["PUSHOVER_USER"],
                         "message": message,
                     }), {"Content-type": "application/x-www-form-urlencoded"})
        conn.getresponse()
