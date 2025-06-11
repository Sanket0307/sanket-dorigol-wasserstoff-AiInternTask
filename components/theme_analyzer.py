import json
import re
import streamlit as st
from typing import List, Dict


class ThemeAnalyzer:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    def analyze_document_themes(self, content, document_name=""):
        """Analyze themes in a single document using Gemini"""
        try:
            # Get theme analysis from Gemini
            response = self.gemini_client.analyze_themes(content, document_name)

            # Parse the JSON response
            themes = self._parse_theme_response(response)

            return themes

        except Exception as e:
            st.error(f"Error analyzing themes for {document_name}: {str(e)}")
            return []

    def _parse_theme_response(self, response_text):
        """Parse theme response from Gemini"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                theme_data = json.loads(json_str)

                if 'themes' in theme_data:
                    return self._validate_themes(theme_data['themes'])

            # Fallback: create basic themes from text
            return self._extract_themes_from_text(response_text)

        except Exception as e:
            st.warning(f"Error parsing theme response: {str(e)}")
            return self._extract_themes_from_text(response_text)

    def _validate_themes(self, themes):
        """Validate and clean theme data"""
        validated_themes = []

        for theme in themes:
            if isinstance(theme, dict) and 'name' in theme:
                validated_theme = {
                    'name': theme.get('name', 'Unnamed Theme'),
                    'description': theme.get('description', 'No description available'),
                    'confidence': min(max(theme.get('confidence', 0.5), 0.0), 1.0),
                    'key_points': theme.get('key_points', [])
                }
                validated_themes.append(validated_theme)

        return validated_themes

    def _extract_themes_from_text(self, text):
        """Fallback method to extract themes from text"""
        themes = []

        # Simple pattern matching for themes
        theme_patterns = [
            r'Theme\s*\d*:?\s*([^\n]+)',
            r'(\d+\.\s*[^\n]+)',
            r'Main theme:?\s*([^\n]+)',
            r'Key theme:?\s*([^\n]+)'
        ]

        found_themes = set()
        for pattern in theme_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                theme_name = match.strip()
                if len(theme_name) > 5 and theme_name not in found_themes:
                    found_themes.add(theme_name)

        # Create basic theme structure
        for theme_name in list(found_themes)[:5]:  # Limit to 5 themes
            themes.append({
                'name': theme_name,
                'description': f'Theme identified from document analysis',
                'confidence': 0.7,
                'key_points': []
            })

        return themes

    def aggregate_themes_across_documents(self, all_document_themes):
        """Aggregate and analyze themes across all documents"""
        try:
            # Flatten all themes
            all_themes = []
            for doc_id, themes in all_document_themes.items():
                all_themes.extend(themes)

            if not all_themes:
                return []

            # Create prompt for cross-document theme analysis
            themes_summary = [
                {
                    'name': theme['name'],
                    'description': theme['description']
                }
                for theme in all_themes
            ]

            prompt = f"""
            Analyze these themes found across multiple documents and identify:
            1. Common themes that appear across documents
            2. Relationships between themes
            3. Overall patterns and insights

            Themes found:
            {json.dumps(themes_summary, indent=2)}

            Return analysis in JSON format:
            {{
                "common_themes": ["theme1", "theme2"],
                "theme_relationships": {{"theme1": ["related_theme1", "related_theme2"]}},
                "insights": "Overall insights about the document collection"
            }}
            """

            response = self.gemini_client.generate_response(prompt)

            # Parse and return the analysis
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

            return {
                "common_themes": list(set(theme['name'] for theme in all_themes)),
                "theme_relationships": {},
                "insights": "Themes identified across the document collection."
            }

        except Exception as e:
            st.error(f"Error aggregating themes: {str(e)}")
            return []
