# components/multi_stage_query_processor.py
import streamlit as st
import json
import re
from typing import List, Dict, Any


class MultiStageQueryProcessor:
    def __init__(self, gemini_client, vector_store):
        self.gemini_client = gemini_client
        self.vector_store = vector_store

    def process_complex_query(self, query, documents):
        """Multi-stage query processing for long documents"""

        # Stage 1: Query classification
        query_type = self._classify_query(query)

        # Stage 2: Initial retrieval
        if hasattr(self.vector_store, 'hybrid_search'):
            initial_results = self.vector_store.hybrid_search(query, max_results=20)
        else:
            initial_results = self.vector_store.search(query, max_results=20)

        # Stage 3: Contextual expansion
        if query_type in ['multi_hop', 'summarization', 'comparison']:
            expanded_results = self._expand_context(query, initial_results)
        else:
            expanded_results = initial_results

        # Stage 4: Answer synthesis
        final_answer = self._synthesize_answer(query, expanded_results, query_type)

        return final_answer, expanded_results

    def _classify_query(self, query):
        """Classify query type for appropriate processing"""
        prompt = f"""
        Classify this query into one of these types:
        - factual: Simple fact-based question
        - multi_hop: Requires information from multiple sections
        - summarization: Asks for summary or overview
        - comparison: Compares multiple concepts/items
        - calculation: Involves numerical computation
        - figure_table: About figures, charts, or tables

        Query: "{query}"

        Return only the classification.
        """

        try:
            response = self.gemini_client.generate_response(prompt)
            classification = response.strip().lower()

            # Validate classification
            valid_types = ['factual', 'multi_hop', 'summarization', 'comparison', 'calculation', 'figure_table']
            if classification in valid_types:
                return classification
            else:
                return 'factual'  # Default fallback

        except Exception as e:
            st.warning(f"Error classifying query: {str(e)}")
            return 'factual'

    def _expand_context(self, query, initial_results):
        """Expand context for complex queries"""
        try:
            # Find related sections
            related_queries = self._generate_related_queries(query)

            expanded_results = list(initial_results)

            for related_query in related_queries:
                if hasattr(self.vector_store, 'hybrid_search'):
                    related_results = self.vector_store.hybrid_search(related_query, max_results=5)
                else:
                    related_results = self.vector_store.search(related_query, max_results=5)
                expanded_results.extend(related_results)

            # Remove duplicates and re-rank
            unique_results = self._deduplicate_results(expanded_results)
            return self._rerank_results(query, unique_results)

        except Exception as e:
            st.warning(f"Error expanding context: {str(e)}")
            return initial_results

    def _generate_related_queries(self, query):
        """Generate related queries for context expansion"""
        try:
            prompt = f"""
            Generate 3-5 related queries that would help provide comprehensive context for answering this question:

            Original Query: "{query}"

            Generate queries that explore:
            1. Background information
            2. Related concepts
            3. Supporting details
            4. Alternative perspectives

            Return as a JSON list of strings:
            ["related query 1", "related query 2", "related query 3"]
            """

            response = self.gemini_client.generate_response(prompt)

            # Parse JSON response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                related_queries = json.loads(json_match.group())
                return related_queries[:5]  # Limit to 5 queries
            else:
                return []

        except Exception as e:
            st.warning(f"Error generating related queries: {str(e)}")
            return []

    def _deduplicate_results(self, results):
        """Remove duplicate results based on document id and content similarity"""
        seen_docs = set()
        seen_content = set()
        unique_results = []

        for result in results:
            doc_id = result.get('document_id', '')
            content = result.get('content', '')

            # Create a simple content hash for deduplication
            content_hash = hash(content[:100])  # Use first 100 chars for comparison

            # Check for duplicates
            doc_chunk_key = f"{doc_id}_{content_hash}"

            if doc_chunk_key not in seen_docs:
                unique_results.append(result)
                seen_docs.add(doc_chunk_key)

        return unique_results

    def _rerank_results(self, query, results):
        """Rerank results based on relevance to the query"""
        try:
            # Simple reranking based on similarity scores and content relevance
            query_words = set(query.lower().split())

            def calculate_relevance(result):
                content = result.get('content', '').lower()
                content_words = set(content.split())

                # Calculate word overlap
                overlap = len(query_words.intersection(content_words))
                overlap_score = overlap / len(query_words) if query_words else 0

                # Combine with existing similarity score
                similarity = result.get('similarity', 0.5)

                # Final relevance score
                return (overlap_score * 0.3) + (similarity * 0.7)

            # Sort by relevance score
            reranked = sorted(results, key=calculate_relevance, reverse=True)
            return reranked

        except Exception as e:
            st.warning(f"Error reranking results: {str(e)}")
            return results

    def _synthesize_answer(self, query, results, query_type):
        """Synthesize a final answer from the results based on query type"""
        try:
            if not results:
                return "I couldn't find relevant information to answer your question."

            # Prepare context from results
            context_parts = []
            for i, result in enumerate(results[:10]):  # Limit to top 10 results
                context_parts.append({
                    'document': result.get('document_name', f'Document {i + 1}'),
                    'content': result.get('content', '')[:500],  # Limit content length
                    'similarity': result.get('similarity', 0.5)
                })

            # Create synthesis prompt based on query type
            synthesis_prompt = self._create_synthesis_prompt(query, query_type, context_parts)

            # Generate synthesized answer
            response = self.gemini_client.generate_response(synthesis_prompt)
            return response

        except Exception as e:
            st.error(f"Error synthesizing answer: {str(e)}")
            return "I encountered an error while synthesizing the answer."

    def _create_synthesis_prompt(self, query, query_type, context_parts):
        """Create appropriate synthesis prompt based on query type"""

        base_context = f"""
        Query: "{query}"
        Query Type: {query_type}

        Available Information:
        {json.dumps(context_parts, indent=2)}
        """

        if query_type == 'summarization':
            instruction = """
            Provide a comprehensive summary that synthesizes information from all available sources.
            Structure your response with clear sections and key points.
            """
        elif query_type == 'comparison':
            instruction = """
            Compare and contrast the different perspectives or items mentioned in the sources.
            Highlight similarities, differences, and provide a balanced analysis.
            """
        elif query_type == 'multi_hop':
            instruction = """
            Connect information from multiple sources to provide a complete answer.
            Show how different pieces of information relate to each other.
            """
        elif query_type == 'calculation':
            instruction = """
            Extract relevant numerical data and perform necessary calculations.
            Show your work and explain the methodology.
            """
        elif query_type == 'figure_table':
            instruction = """
            Focus on information from tables, figures, and visual elements.
            Describe the data and its implications clearly.
            """
        else:  # factual
            instruction = """
            Provide a direct, factual answer based on the available information.
            Include specific details and evidence from the sources.
            """

        return base_context + "\n" + instruction + "\n\nProvide a comprehensive answer with proper citations:"

    def get_processing_stats(self, query, results, query_type):
        """Get statistics about the query processing"""
        stats = {
            'query': query,
            'query_type': query_type,
            'total_results': len(results),
            'unique_documents': len(set(r.get('document_id', '') for r in results)),
            'avg_similarity': sum(r.get('similarity', 0) for r in results) / len(results) if results else 0,
            'processing_stages': 4
        }
        return stats
