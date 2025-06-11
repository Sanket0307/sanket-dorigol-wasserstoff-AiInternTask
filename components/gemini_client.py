import os
from google import genai
import streamlit as st


class GeminiClient:
    def __init__(self):
        """Initialize Gemini client"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("❌ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
            st.stop()

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    def generate_response(self, prompt):
        """Generate response from Gemini"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def analyze_themes(self, content, document_name=""):
        """Analyze themes in document content"""
        prompt = f"""
        Analyze the following document content and identify the main themes. 
        Document: {document_name}

        Content:
        {content}

        Please identify 3-7 main themes and return them in JSON format:
        {{
            "themes": [
                {{
                    "name": "Theme Name",
                    "description": "Brief description of the theme",
                    "confidence": 0.0-1.0,
                    "key_points": ["point1", "point2", "point3"]
                }}
            ]
        }}

        Focus on the most significant and recurring themes in the content.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            st.error(f"Error analyzing themes: {str(e)}")
            return '{"themes": []}'
