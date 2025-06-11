from google import genai

client = genai.Client(api_key="AIzaSyDPpMa4TOao7wWoiFCIrwwn7Q18u5XjJTg")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain me what is machine learning"
)
print(response.text)