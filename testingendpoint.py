# Import necessary libraries
import requests
 
# Define the URL of the deployed endpoint
url = "http://127.0.0.1:8000"
 
def test_ner_service(input_text):
    """
    Function to send a POST request to the NER service and print the results.
    """
    response = requests.post(url, data={"text": input_text})
    if response.status_code == 200:
        print("Response received successfully.")
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('li')
        for result in results:
            print(result.text)
    else:
        print(f"Failed to get response. Status code: {response.status_code}")
 
# Test the service with some example text
test_ner_service("Polyethylene terephthalate")