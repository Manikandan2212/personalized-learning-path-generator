"""
Resource Agent
Fetches curated learning resources for each topic using RAG retrieval.
"""

import logging
from typing import List, Dict

logger = logging.getLogger("agents.resource")

CURATED_RESOURCES = {
    "python_intro": [
        {"title": "Python Official Tutorial", "url": "https://docs.python.org/3/tutorial/", "type": "documentation", "free": True, "estimated_hours": 8},
        {"title": "Automate the Boring Stuff", "url": "https://automatetheboringstuff.com/", "type": "book", "free": True, "estimated_hours": 15},
        {"title": "Python for Everybody - Coursera", "url": "https://www.coursera.org/specializations/python", "type": "course", "free": False, "estimated_hours": 30},
    ],
    "linear_algebra": [
        {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab", "type": "video", "free": True, "estimated_hours": 4},
        {"title": "Khan Academy Linear Algebra", "url": "https://www.khanacademy.org/math/linear-algebra", "type": "course", "free": True, "estimated_hours": 20},
        {"title": "MIT 18.06 Linear Algebra", "url": "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", "type": "course", "free": True, "estimated_hours": 40},
    ],
    "calculus": [
        {"title": "3Blue1Brown: Essence of Calculus", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr", "type": "video", "free": True, "estimated_hours": 3},
        {"title": "Khan Academy Calculus", "url": "https://www.khanacademy.org/math/calculus-1", "type": "course", "free": True, "estimated_hours": 25},
    ],
    "statistics_basics": [
        {"title": "StatQuest with Josh Starmer", "url": "https://www.youtube.com/@statquest", "type": "video", "free": True, "estimated_hours": 10},
        {"title": "Think Stats (Free Book)", "url": "https://greenteapress.com/thinkstats2/", "type": "book", "free": True, "estimated_hours": 15},
        {"title": "Khan Academy Statistics", "url": "https://www.khanacademy.org/math/statistics-probability", "type": "course", "free": True, "estimated_hours": 20},
    ],
    "pandas_numpy": [
        {"title": "Pandas Official Docs", "url": "https://pandas.pydata.org/docs/getting_started/intro_tutorials/", "type": "documentation", "free": True, "estimated_hours": 5},
        {"title": "Kaggle Pandas Course", "url": "https://www.kaggle.com/learn/pandas", "type": "course", "free": True, "estimated_hours": 4},
        {"title": "NumPy Quickstart", "url": "https://numpy.org/doc/stable/user/quickstart.html", "type": "documentation", "free": True, "estimated_hours": 2},
    ],
    "ml_foundations": [
        {"title": "Google ML Crash Course", "url": "https://developers.google.com/machine-learning/crash-course", "type": "course", "free": True, "estimated_hours": 15},
        {"title": "Kaggle Intro to ML", "url": "https://www.kaggle.com/learn/intro-to-machine-learning", "type": "course", "free": True, "estimated_hours": 3},
        {"title": "Scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html", "type": "documentation", "free": True, "estimated_hours": 10},
        {"title": "Hands-On ML (O'Reilly)", "url": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/", "type": "book", "free": False, "estimated_hours": 40},
    ],
    "neural_networks": [
        {"title": "Neural Networks and Deep Learning (Free Book)", "url": "http://neuralnetworksanddeeplearning.com/", "type": "book", "free": True, "estimated_hours": 15},
        {"title": "Andrej Karpathy: Neural Nets Zero to Hero", "url": "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ", "type": "video", "free": True, "estimated_hours": 8},
    ],
    "deep_learning": [
        {"title": "Deep Learning Specialization (Coursera)", "url": "https://www.coursera.org/specializations/deep-learning", "type": "course", "free": False, "estimated_hours": 80},
        {"title": "Fast.ai Practical Deep Learning", "url": "https://course.fast.ai/", "type": "course", "free": True, "estimated_hours": 30},
        {"title": "Deep Learning Book (Goodfellow)", "url": "https://www.deeplearningbook.org/", "type": "book", "free": True, "estimated_hours": 40},
    ],
    "transformers_nlp": [
        {"title": "HuggingFace NLP Course", "url": "https://huggingface.co/course/", "type": "course", "free": True, "estimated_hours": 20},
        {"title": "Attention Is All You Need (Paper)", "url": "https://arxiv.org/abs/1706.03762", "type": "paper", "free": True, "estimated_hours": 2},
        {"title": "Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/", "type": "article", "free": True, "estimated_hours": 1},
    ],
    "rag_systems": [
        {"title": "LangChain RAG Tutorial", "url": "https://python.langchain.com/docs/tutorials/rag/", "type": "documentation", "free": True, "estimated_hours": 5},
        {"title": "RAG Survey Paper", "url": "https://arxiv.org/abs/2312.10997", "type": "paper", "free": True, "estimated_hours": 3},
    ],
    "prompt_engineering": [
        {"title": "Anthropic Prompt Engineering Guide", "url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview", "type": "documentation", "free": True, "estimated_hours": 3},
        {"title": "OpenAI Prompt Engineering Guide", "url": "https://platform.openai.com/docs/guides/prompt-engineering", "type": "documentation", "free": True, "estimated_hours": 2},
        {"title": "LearnPrompting.org", "url": "https://learnprompting.org/", "type": "course", "free": True, "estimated_hours": 5},
    ],
    "html_css": [
        {"title": "MDN Web Docs - HTML", "url": "https://developer.mozilla.org/en-US/docs/Learn/HTML", "type": "documentation", "free": True, "estimated_hours": 10},
        {"title": "CSS Tricks", "url": "https://css-tricks.com/", "type": "article", "free": True, "estimated_hours": 5},
        {"title": "freeCodeCamp Responsive Web Design", "url": "https://www.freecodecamp.org/learn/2022/responsive-web-design/", "type": "course", "free": True, "estimated_hours": 20},
    ],
    "javascript": [
        {"title": "javascript.info", "url": "https://javascript.info/", "type": "book", "free": True, "estimated_hours": 30},
        {"title": "MDN JavaScript Guide", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide", "type": "documentation", "free": True, "estimated_hours": 15},
        {"title": "freeCodeCamp JavaScript", "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/", "type": "course", "free": True, "estimated_hours": 30},
    ],
    "python_oop": [
        {"title": "Real Python - OOP in Python", "url": "https://realpython.com/python3-object-oriented-programming/", "type": "article", "free": True, "estimated_hours": 3},
        {"title": "Python Docs - Classes", "url": "https://docs.python.org/3/tutorial/classes.html", "type": "documentation", "free": True, "estimated_hours": 2},
    ],
    "data_visualization": [
        {"title": "Matplotlib Tutorials", "url": "https://matplotlib.org/stable/tutorials/index.html", "type": "documentation", "free": True, "estimated_hours": 5},
        {"title": "Kaggle Data Visualization", "url": "https://www.kaggle.com/learn/data-visualization", "type": "course", "free": True, "estimated_hours": 4},
    ],
}


class ResourceAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.name = "ResourceAgent"

    def fetch_resources(self, topic_id: str, mcp_context: Dict) -> Dict:
        logger.info(f"ResourceAgent: fetching resources for topic='{topic_id}'")

        # Get curated resources
        curated = CURATED_RESOURCES.get(topic_id, [])

        # RAG search to find related context
        query = topic_id.replace("_", " ")
        retrieved = self.vector_store.search(query, top_k=3, min_score=0.01)

        related_topics = []
        seen = set()
        for r in retrieved:
            src = r["metadata"]["source_id"]
            if src != topic_id and src not in seen:
                seen.add(src)
                related_topics.append({
                    "topic_id": src,
                    "title": src.replace("_", " ").title(),
                    "relevance_score": r["score"],
                })

        free_count = sum(1 for r in curated if r.get("free", True))
        total_hours = sum(r.get("estimated_hours", 0) for r in curated)

        result = {
            "topic_id": topic_id,
            "title": topic_id.replace("_", " ").title(),
            "resources": curated,
            "related_topics": related_topics[:3],
            "summary": {
                "total_resources": len(curated),
                "free_resources": free_count,
                "paid_resources": len(curated) - free_count,
                "estimated_total_hours": total_hours,
            },
            "mcp_context": {
                "user_goal": mcp_context.get("user_goal"),
                "current_step": mcp_context.get("current_step"),
            },
        }

        logger.info(f"ResourceAgent: found {len(curated)} resources for '{topic_id}'")
        return result

    def search_resources(self, query: str, mcp_context: Dict) -> Dict:
        logger.info(f"ResourceAgent: searching resources for query='{query}'")
        retrieved = self.vector_store.search(query, top_k=6, min_score=0.0)

        results = []
        seen_sources = set()
        for r in retrieved:
            src = r["metadata"]["source_id"]
            if src in seen_sources:
                continue
            seen_sources.add(src)
            resources = CURATED_RESOURCES.get(src, [])
            if resources:
                results.append({
                    "topic_id": src,
                    "title": src.replace("_", " ").title(),
                    "relevance_score": r["score"],
                    "level": r["metadata"].get("level", "unknown"),
                    "top_resource": resources[0] if resources else None,
                    "resource_count": len(resources),
                })

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "mcp_context": {"user_goal": mcp_context.get("user_goal")},
        }
