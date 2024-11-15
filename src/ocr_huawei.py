from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkcore.http.http_config import HttpConfig
# Import the huaweicloudsdk{service} library of a specified cloud service.
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkocr.v1 import *
import os

# Use the default configuration. If the error message "'HttpConfig' is not defined" is displayed, check whether the SDK is installed correctly.
# config = HttpConfig.get_default_config()


# The default connection timeout interval is 60 seconds, and the read timeout interval is 120 seconds. You can set timeout to timeout or (connect timeout, read timeout).
# config.timeout = 120

# ak = os.environ.get("HUAWEICLOUD_SDK_AK")
# sk = os.environ.get("HUAWEICLOUD_SDK_SK")
# project_id= os.environ.get("PROJECT_ID")
# security_token = os.environ.get("TOKEN")

# credentials = BasicCredentials(ak, sk, project_id)

# credentials = BasicCredentials(ak, sk, project_id).with_security_token(security_token)



import requests
import json
import base64
import os
from dotenv import load_dotenv

load_dotenv()

class HuaweiOCRClient:
    def __init__(self):
        self.auth_url = "https://iam.ap-southeast-2.myhuaweicloud.com/v3/auth/tokens"
        self.ocr_url_template = "https://ocr.ap-southeast-2.myhuaweicloud.com/v2/{project_id}/ocr/general-text"
        self.domain_user = os.getenv('DOMAIN_USER')
        self.iam_user = os.getenv('IAM_USER')
        self.password = os.getenv('PASSWORD')
        self.security_token = None
        self.project_id = None
        self.authenticate()

    def authenticate(self):
        payload = json.dumps({
            "auth": {
                "identity": {
                    "methods": ["password"],
                    "password": {
                        "user": {
                            "name": self.iam_user,
                            "password": self.password,
                            "domain": {"name": self.domain_user}
                        }
                    }
                },
                "scope": {"project": {"name": "ap-southeast-2"}}
            }
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.auth_url, headers=headers, data=payload)
        
        if response.status_code == 201:
            self.security_token = response.headers['X-Subject-Token']
            self.project_id = response.json()['token']['project']['domain']['id']
            print("Authentication successful")
        else:
            raise Exception("Authentication failed")

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def perform_ocr(self, image_path, language="th", quick_mode=False, detect_direction=True):
        if not self.security_token or not self.project_id:
            raise Exception("Client is not authenticated. Please authenticate first.")
        
        base64_image = self.encode_image_to_base64(image_path)
        payload = json.dumps({
            "image": base64_image,
            "language": language,
            "quick_mode": quick_mode,
            "detect_direction": detect_direction
        })
        headers = {
            'X-Auth-Token': self.security_token,
            'Content-Type': 'application/json'
        }
        ocr_url = self.ocr_url_template.format(project_id=self.project_id)
        response = requests.post(ocr_url, headers=headers, data=payload)

        if response.status_code == 200:
            return response.json().get("result", {}).get("words_block_list", [])
        else:
            raise Exception("OCR request failed")

if __name__ == "__main__":
    client = HuaweiOCRClient()
    word_container = client.perform_ocr("./test/5_page_1.jpg")
    print(word_container)

if __name__ == "__main__":
    client = HuaweiOCRClient()
    try:
        word_container = client.perform_ocr("./test/5_page_1.jpg")
        
        # Convert the word blocks to a plain text format
        result_text = "\n".join(block["words"] for block in word_container if "words" in block)
        
        # Print the result
        print(result_text)
        
    except Exception as e:
        # logging.error(e)
        print(e)






