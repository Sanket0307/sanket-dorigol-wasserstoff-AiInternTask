# components/document_segmenter.py
import json
import re
import streamlit as st
from typing import List, Dict, Any


class IntelligentSegmenter:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    def segment_long_document(self, content, doc_name):
        """Intelligently segment long documents"""
        try:
            # Detect document structure
            structure = self._detect_document_structure(content)

            # Create logical segments
            segments = self._create_logical_segments(content, structure)

            # Generate segment summaries
            segment_summaries = []
            for segment in segments:
                summary = self._generate_segment_summary(segment, doc_name)
                segment_summaries.append({
                    'content': segment['content'],
                    'summary': summary,
                    'section_type': segment['type'],
                    'importance_score': segment['importance'],
                    'start_pos': segment['start_pos'],
                    'end_pos': segment['end_pos']
                })

            return segment_summaries

        except Exception as e:
            st.error(f"Error in document segmentation: {str(e)}")
            return self._fallback_segmentation(content, doc_name)

    def _detect_document_structure(self, content):
        """Detect document structure using AI"""
        prompt = f"""
        Analyze this document content and identify its structure:

        {content[:2000]}...

        Identify:
        1. Document type (research paper, legal document, manual, etc.)
        2. Main sections (abstract, introduction, methodology, etc.)
        3. Hierarchical structure (headings, subheadings)
        4. Special elements (tables, figures, appendices)

        Return as JSON with section boundaries and types:
        {{
            "document_type": "type_here",
            "sections": [
                {{
                    "name": "section_name",
                    "type": "section_type",
                    "start": 0,
                    "end": 1000,
                    "level": 1
                }}
            ]
        }}
        """

        try:
            response = self.gemini_client.generate_response(prompt)
            return self._parse_structure_response(response)
        except Exception as e:
            st.warning(f"Error detecting document structure: {str(e)}")
            return self._fallback_structure_detection(content)

    def _parse_structure_response(self, response):
        """Parse the JSON structure response from AI"""
        try:
            # Extract JSON from response text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                structure = json.loads(json_str)

                # Validate structure
                if 'sections' in structure and isinstance(structure['sections'], list):
                    return structure
                else:
                    return {'sections': []}
            else:
                return {'sections': []}
        except Exception as e:
            st.warning(f"Error parsing structure response: {str(e)}")
            return {'sections': []}

    def _create_logical_segments(self, content, structure):
        """Create logical segments based on structure"""
        segments = []

        sections = structure.get('sections', [])

        if not sections:
            # If no structure detected, create basic segments
            return self._create_basic_segments(content)

        for section in sections:
            try:
                start_pos = section.get('start', 0)
                end_pos = section.get('end', len(content))

                # Ensure valid boundaries
                start_pos = max(0, min(start_pos, len(content)))
                end_pos = max(start_pos, min(end_pos, len(content)))

                segment_content = content[start_pos:end_pos]

                if segment_content.strip():  # Only add non-empty segments
                    # Calculate importance score
                    importance = self._calculate_importance(segment_content, section.get('type', 'unknown'))

                    segments.append({
                        'content': segment_content,
                        'type': section.get('type', 'unknown'),
                        'name': section.get('name', 'Unnamed Section'),
                        'importance': importance,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'level': section.get('level', 1)
                    })
            except Exception as e:
                st.warning(f"Error creating segment: {str(e)}")
                continue

        return segments if segments else self._create_basic_segments(content)

    def _create_basic_segments(self, content):
        """Create basic segments when structure detection fails"""
        segments = []
        segment_size = 2000  # Characters per segment
        overlap = 200

        for i in range(0, len(content), segment_size - overlap):
            start_pos = i
            end_pos = min(i + segment_size, len(content))
            segment_content = content[start_pos:end_pos]

            if segment_content.strip():
                segments.append({
                    'content': segment_content,
                    'type': 'basic_segment',
                    'name': f'Segment {len(segments) + 1}',
                    'importance': 0.5,  # Default importance
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'level': 1
                })

        return segments

    def _calculate_importance(self, segment_content, segment_type):
        """Calculate importance score for a segment"""
        try:
            # Base score based on content length
            base_score = min(1.0, len(segment_content) / 2000.0)

            # Type-based weighting
            type_weights = {
                'abstract': 1.5,
                'summary': 1.5,
                'conclusion': 1.4,
                'introduction': 1.3,
                'discussion': 1.2,
                'results': 1.2,
                'methodology': 1.1,
                'literature_review': 1.0,
                'background': 1.0,
                'appendix': 0.5,
                'references': 0.3,
                'bibliography': 0.3,
                'unknown': 0.8,
                'basic_segment': 0.7
            }

            type_weight = type_weights.get(segment_type.lower(), 1.0)

            # Content-based scoring
            content_score = self._analyze_content_importance(segment_content)

            # Combined importance score
            importance = (base_score * 0.4 + type_weight * 0.4 + content_score * 0.2)

            # Normalize to 0-1 range
            return min(1.0, max(0.0, importance))

        except Exception as e:
            return 0.5  # Default importance if calculation fails

    def _analyze_content_importance(self, content):
        """Analyze content to determine importance"""
        try:
            # Keywords that indicate important content
            important_keywords = [
                'conclusion', 'summary', 'findings', 'results', 'key', 'important',
                'significant', 'main', 'primary', 'critical', 'essential', 'major'
            ]

            # Keywords that indicate less important content
            less_important_keywords = [
                'appendix', 'reference', 'bibliography', 'footnote', 'citation'
            ]

            content_lower = content.lower()

            # Count important keywords
            important_count = sum(1 for keyword in important_keywords if keyword in content_lower)
            less_important_count = sum(1 for keyword in less_important_keywords if keyword in content_lower)

            # Calculate score based on keyword presence
            keyword_score = (important_count - less_important_count) / max(1, len(content.split()) / 100)

            # Normalize to 0-1 range
            return min(1.0, max(0.0, keyword_score + 0.5))

        except Exception as e:
            return 0.5

    def _generate_segment_summary(self, segment, doc_name):
        """Generate summary for a document segment"""
        try:
            content = segment['content']
            segment_type = segment.get('type', 'unknown')

            # Limit content length for API call
            content_preview = content[:1500] + "..." if len(content) > 1500 else content

            prompt = f"""
            Generate a concise summary for this document segment:

            Document: {doc_name}
            Section Type: {segment_type}
            Content: {content_preview}

            Provide a 2-3 sentence summary that captures the main points of this segment.
            Focus on the key information and insights.
            """

            response = self.gemini_client.generate_response(prompt)
            return response.strip()

        except Exception as e:
            st.warning(f"Error generating segment summary: {str(e)}")
            return f"Summary unavailable for {segment.get('type', 'unknown')} segment."

    def _fallback_structure_detection(self, content):
        """Fallback method for structure detection"""
        try:
            # Simple heuristic-based structure detection
            lines = content.split('\n')
            sections = []
            current_pos = 0

            # Look for potential headings (lines with fewer words, title case, etc.)
            for i, line in enumerate(lines):
                line = line.strip()
                if self._is_potential_heading(line):
                    if sections:  # Close previous section
                        sections[-1]['end'] = current_pos

                    # Start new section
                    sections.append({
                        'name': line,
                        'type': self._classify_section_type(line),
                        'start': current_pos,
                        'end': len(content),  # Will be updated
                        'level': 1
                    })

                current_pos += len(line) + 1  # +1 for newline

            return {'sections': sections}

        except Exception as e:
            return {'sections': []}

    def _is_potential_heading(self, line):
        """Determine if a line is likely a heading"""
        if not line or len(line) > 100:  # Too long to be a heading
            return False

        # Check for heading indicators
        words = line.split()
        if len(words) > 10:  # Too many words for a heading
            return False

        # Check for title case or all caps
        if line.isupper() or line.istitle():
            return True

        # Check for numbered sections
        if re.match(r'^\d+\.?\s+', line):
            return True

        # Check for common section headers
        common_headers = [
            'abstract', 'introduction', 'methodology', 'results', 'discussion',
            'conclusion', 'references', 'appendix', 'summary', 'background'
        ]

        if any(header in line.lower() for header in common_headers):
            return True

        return False

    def _classify_section_type(self, heading):
        """Classify section type based on heading text"""
        heading_lower = heading.lower()

        type_mapping = {
            'abstract': 'abstract',
            'summary': 'summary',
            'introduction': 'introduction',
            'background': 'background',
            'methodology': 'methodology',
            'method': 'methodology',
            'results': 'results',
            'findings': 'results',
            'discussion': 'discussion',
            'conclusion': 'conclusion',
            'references': 'references',
            'bibliography': 'bibliography',
            'appendix': 'appendix'
        }

        for keyword, section_type in type_mapping.items():
            if keyword in heading_lower:
                return section_type

        return 'unknown'

    def _fallback_segmentation(self, content, doc_name):
        """Fallback segmentation when AI processing fails"""
        try:
            segments = self._create_basic_segments(content)

            segment_summaries = []
            for segment in segments:
                segment_summaries.append({
                    'content': segment['content'],
                    'summary': f"Segment from {doc_name} ({len(segment['content'])} characters)",
                    'section_type': segment['type'],
                    'importance_score': segment['importance'],
                    'start_pos': segment['start_pos'],
                    'end_pos': segment['end_pos']
                })

            return segment_summaries

        except Exception as e:
            st.error(f"Fallback segmentation failed: {str(e)}")
            return []

    def get_segmentation_stats(self, segments):
        """Get statistics about the segmentation"""
        if not segments:
            return {}

        stats = {
            'total_segments': len(segments),
            'avg_segment_length': sum(len(seg['content']) for seg in segments) / len(segments),
            'section_types': list(set(seg['section_type'] for seg in segments)),
            'high_importance_segments': len([seg for seg in segments if seg['importance_score'] > 0.7]),
            'total_content_length': sum(len(seg['content']) for seg in segments)
        }

        return stats
