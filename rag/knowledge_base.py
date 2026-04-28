"""
Knowledge Base
Seeds the vector store with learning content across multiple domains.
"""
from typing import Optional, Dict, List
from rag.vector_store import VectorStore, chunk_text
import logging

logger = logging.getLogger("rag.knowledge_base")

KNOWLEDGE_BASE = [
    # ── PYTHON ──────────────────────────────────────────────────────────
    {
        "id": "python_intro",
        "topic": "Python",
        "level": "beginner",
        "content": """Python is a high-level, interpreted programming language known for its clear syntax and readability.
        It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        Python uses indentation to define code blocks instead of curly braces. Variables do not need explicit type
        declarations. Python has a large standard library and an enormous ecosystem of third-party packages.
        Key features: dynamic typing, automatic memory management, list comprehensions, generators, decorators,
        and context managers. Python is widely used in web development, data science, automation, and AI.""",
        "prerequisites": [],
        "resources": [
            {"title": "Python Official Docs", "url": "https://docs.python.org/3/"},
            {"title": "Python Tutorial", "url": "https://docs.python.org/3/tutorial/"},
        ],
        "duration_hours": 10,
    },
    {
        "id": "python_oop",
        "topic": "Python",
        "level": "intermediate",
        "content": """Object-Oriented Programming in Python uses classes and objects. A class is a blueprint for creating
        objects. Key OOP concepts: encapsulation (bundling data and methods), inheritance (child classes extend parent
        classes), polymorphism (same interface, different implementations), and abstraction (hiding implementation
        details). In Python, you define classes with the class keyword. The __init__ method is the constructor.
        Self refers to the instance. Python supports multiple inheritance. Magic methods (dunder methods) allow
        customization of built-in operations. Abstract base classes enforce interface contracts.""",
        "prerequisites": ["python_intro"],
        "resources": [
            {"title": "Python Classes", "url": "https://docs.python.org/3/tutorial/classes.html"},
        ],
        "duration_hours": 15,
    },
    # ── MACHINE LEARNING ────────────────────────────────────────────────
    {
        "id": "ml_foundations",
        "topic": "Machine Learning",
        "level": "beginner",
        "content": """Machine Learning is a subset of artificial intelligence where systems learn from data to make
        predictions or decisions without being explicitly programmed. The three main types are supervised learning
        (labeled data, predicts output), unsupervised learning (unlabeled data, finds patterns), and reinforcement
        learning (learns from rewards and penalties). Key concepts include features (input variables), labels
        (output variables), training data, validation data, test data, overfitting (model memorizes training data),
        underfitting (model too simple), bias-variance tradeoff, and cross-validation. Common algorithms include
        linear regression, logistic regression, decision trees, random forests, SVM, and k-nearest neighbors.""",
        "prerequisites": ["linear_algebra", "statistics_basics"],
        "resources": [
            {"title": "Scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html"},
            {"title": "ML Crash Course", "url": "https://developers.google.com/machine-learning/crash-course"},
        ],
        "duration_hours": 30,
    },
    {
        "id": "deep_learning",
        "topic": "Deep Learning",
        "level": "advanced",
        "content": """Deep Learning uses neural networks with many layers to learn representations from data.
        A neural network consists of an input layer, hidden layers, and an output layer. Each neuron applies
        a weighted sum and an activation function. Training uses backpropagation and gradient descent to minimize
        a loss function. Key architectures: Convolutional Neural Networks (CNNs) for image tasks using filters
        and pooling; Recurrent Neural Networks (RNNs) for sequential data; Transformers for NLP using
        self-attention mechanisms. Regularization techniques: dropout, batch normalization, L1/L2 regularization.
        Frameworks: PyTorch and TensorFlow. GPU acceleration is essential for large models.""",
        "prerequisites": ["ml_foundations", "linear_algebra", "calculus"],
        "resources": [
            {"title": "Deep Learning Book", "url": "https://www.deeplearningbook.org/"},
            {"title": "Fast.ai Course", "url": "https://course.fast.ai/"},
        ],
        "duration_hours": 60,
    },
    {
        "id": "neural_networks",
        "topic": "Neural Networks",
        "level": "intermediate",
        "content": """Neural networks are composed of layers of neurons. The forward pass computes predictions:
        input → hidden layers → output. Activation functions introduce non-linearity: ReLU (max(0,x)) is most
        common in hidden layers. Sigmoid and Softmax are used in output layers for classification. Loss functions
        measure prediction error: Mean Squared Error for regression, Cross-Entropy for classification.
        Backpropagation computes gradients using the chain rule. Optimizers update weights: SGD (stochastic
        gradient descent), Adam (adaptive moment estimation), RMSProp. Hyperparameters: learning rate, batch
        size, number of epochs, network depth and width. Initialization matters: Xavier/Glorot initialization
        prevents vanishing/exploding gradients.""",
        "prerequisites": ["ml_foundations", "calculus", "linear_algebra"],
        "resources": [
            {"title": "Neural Networks and Deep Learning", "url": "http://neuralnetworksanddeeplearning.com/"},
        ],
        "duration_hours": 25,
    },
    # ── MATHEMATICS ─────────────────────────────────────────────────────
    {
        "id": "linear_algebra",
        "topic": "Mathematics",
        "level": "beginner",
        "content": """Linear Algebra is foundational for machine learning and data science. Core concepts:
        scalars (single numbers), vectors (ordered lists of numbers), matrices (2D arrays of numbers), and
        tensors (multi-dimensional arrays). Operations: vector addition, scalar multiplication, dot product
        (measures similarity between vectors), matrix multiplication, transpose. Concepts: eigenvalues and
        eigenvectors (fundamental for PCA and SVD), matrix determinant, matrix inverse, rank. Applications
        in ML: weight matrices in neural networks, covariance matrices in statistics, dimensionality reduction
        via PCA, attention in transformers uses dot product of query and key vectors. NumPy is the standard
        Python library for linear algebra operations.""",
        "prerequisites": [],
        "resources": [
            {"title": "3Blue1Brown Linear Algebra", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"},
            {"title": "Khan Academy Linear Algebra", "url": "https://www.khanacademy.org/math/linear-algebra"},
        ],
        "duration_hours": 20,
    },
    {
        "id": "calculus",
        "topic": "Mathematics",
        "level": "beginner",
        "content": """Calculus is essential for understanding optimization in machine learning. Differential calculus:
        derivatives measure the rate of change of a function. The chain rule is critical for backpropagation.
        Partial derivatives measure change with respect to one variable. Gradient is the vector of all partial
        derivatives. Integral calculus: integrals compute the area under a curve, used in probability distributions.
        Gradient descent uses derivatives to minimize loss functions: move opposite to gradient direction.
        Key concepts for ML: local/global minima, saddle points, learning rate, convex vs non-convex optimization,
        second-order methods (Newton's method). Numerical differentiation vs automatic differentiation (autograd
        in PyTorch).""",
        "prerequisites": [],
        "resources": [
            {"title": "3Blue1Brown Calculus", "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr"},
        ],
        "duration_hours": 20,
    },
    {
        "id": "statistics_basics",
        "topic": "Mathematics",
        "level": "beginner",
        "content": """Statistics provides the mathematical foundations for data science and machine learning.
        Descriptive statistics: mean, median, mode, variance, standard deviation, percentiles, histograms.
        Probability theory: random variables, probability distributions (Normal/Gaussian, Bernoulli, Binomial,
        Poisson), conditional probability, Bayes theorem. Inferential statistics: hypothesis testing, p-values,
        confidence intervals, t-tests, chi-square tests. Regression analysis: linear regression, correlation vs
        causation. For ML: maximum likelihood estimation, Bayesian inference, expectation-maximization,
        information theory (entropy, KL divergence), central limit theorem. Tools: scipy.stats, statsmodels.""",
        "prerequisites": [],
        "resources": [
            {"title": "Statistics and Probability - Khan Academy", "url": "https://www.khanacademy.org/math/statistics-probability"},
        ],
        "duration_hours": 25,
    },
    # ── DATA SCIENCE ────────────────────────────────────────────────────
    {
        "id": "pandas_numpy",
        "topic": "Data Science",
        "level": "beginner",
        "content": """NumPy provides fast multi-dimensional arrays and mathematical operations. Arrays are faster
        than Python lists for numerical computation. Broadcasting allows operations on arrays of different shapes.
        Key NumPy functions: np.array, np.zeros, np.ones, np.arange, np.reshape, np.concatenate, np.dot,
        np.linalg for linear algebra. Pandas builds on NumPy for data manipulation. Key structures: Series
        (1D labeled array) and DataFrame (2D table). Operations: reading CSV/JSON, filtering rows, selecting
        columns, handling missing values (dropna, fillna), groupby aggregations, merging DataFrames, pivot tables,
        time series operations. Data cleaning: removing duplicates, type conversion, string operations.
        Pandas is essential for exploratory data analysis (EDA).""",
        "prerequisites": ["python_intro"],
        "resources": [
            {"title": "Pandas Documentation", "url": "https://pandas.pydata.org/docs/"},
            {"title": "NumPy Quickstart", "url": "https://numpy.org/doc/stable/user/quickstart.html"},
        ],
        "duration_hours": 15,
    },
    {
        "id": "data_visualization",
        "topic": "Data Science",
        "level": "intermediate",
        "content": """Data visualization communicates insights from data visually. Python libraries: Matplotlib
        (foundational, low-level control), Seaborn (statistical plots, built on Matplotlib), Plotly (interactive
        charts), and Altair (declarative grammar). Chart types: line charts (trends over time), bar charts
        (comparisons), scatter plots (relationships between variables), histograms (distributions), box plots
        (quartiles and outliers), heatmaps (correlation matrices), pair plots (relationships between all features).
        Best practices: choose appropriate chart type, label axes clearly, avoid chartjunk, use color meaningfully,
        tell a story with data. EDA (Exploratory Data Analysis) uses visualization to understand data before
        modeling.""",
        "prerequisites": ["pandas_numpy"],
        "resources": [
            {"title": "Matplotlib Tutorials", "url": "https://matplotlib.org/stable/tutorials/index.html"},
            {"title": "Seaborn Tutorial", "url": "https://seaborn.pydata.org/tutorial.html"},
        ],
        "duration_hours": 10,
    },
    # ── WEB DEVELOPMENT ─────────────────────────────────────────────────
    {
        "id": "html_css",
        "topic": "Web Development",
        "level": "beginner",
        "content": """HTML (HyperText Markup Language) structures web content using elements and tags. Key elements:
        headings (h1-h6), paragraphs (p), links (a), images (img), lists (ul, ol, li), divs and spans for layout,
        forms (form, input, button, select), semantic elements (header, nav, main, footer, article, section).
        CSS (Cascading Style Sheets) styles HTML elements. The box model: content, padding, border, margin.
        Selectors: element, class (.classname), ID (#idname), pseudo-classes (:hover, :focus). Layout: Flexbox
        (one-dimensional), CSS Grid (two-dimensional), positioning (static, relative, absolute, fixed).
        Responsive design uses media queries to adapt to different screen sizes. CSS variables for design systems.""",
        "prerequisites": [],
        "resources": [
            {"title": "MDN HTML", "url": "https://developer.mozilla.org/en-US/docs/Web/HTML"},
            {"title": "MDN CSS", "url": "https://developer.mozilla.org/en-US/docs/Web/CSS"},
        ],
        "duration_hours": 20,
    },
    {
        "id": "javascript",
        "topic": "Web Development",
        "level": "beginner",
        "content": """JavaScript is the programming language of the web, running in browsers and on servers (Node.js).
        Core concepts: variables (let, const, var), data types (string, number, boolean, null, undefined, object,
        symbol), functions (declarations, expressions, arrow functions), control flow (if/else, for, while, switch),
        arrays and objects, DOM manipulation (document.querySelector, addEventListener, innerHTML). ES6+ features:
        destructuring, spread/rest operators, template literals, modules (import/export), classes, Promises,
        async/await for asynchronous programming. Event loop, call stack, and the microtask queue. Fetch API for
        HTTP requests. JSON for data interchange. Browser APIs: localStorage, sessionStorage, geolocation.""",
        "prerequisites": ["html_css"],
        "resources": [
            {"title": "JavaScript MDN", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript"},
            {"title": "javascript.info", "url": "https://javascript.info/"},
        ],
        "duration_hours": 30,
    },
    # ── AI/LLM ──────────────────────────────────────────────────────────
    {
        "id": "transformers_nlp",
        "topic": "NLP",
        "level": "advanced",
        "content": """Transformers revolutionized natural language processing with the attention mechanism introduced
        in the paper 'Attention Is All You Need' (2017). Self-attention allows each token to attend to all other
        tokens, capturing long-range dependencies. Multi-head attention applies multiple attention operations
        in parallel, capturing different relationship types. The encoder processes input, the decoder generates
        output. Key models: BERT (bidirectional encoder, good for classification and Q&A), GPT (decoder-only,
        autoregressive text generation), T5 (encoder-decoder, text-to-text). Pre-training on large corpora then
        fine-tuning on downstream tasks. The HuggingFace library provides thousands of pretrained models.
        Tokenization: subword tokenization using BPE or WordPiece.""",
        "prerequisites": ["deep_learning", "neural_networks"],
        "resources": [
            {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762"},
            {"title": "HuggingFace Course", "url": "https://huggingface.co/course/"},
        ],
        "duration_hours": 40,
    },
    {
        "id": "rag_systems",
        "topic": "AI Engineering",
        "level": "advanced",
        "content": """Retrieval Augmented Generation (RAG) combines retrieval systems with language model generation.
        RAG addresses knowledge cutoffs and hallucination in LLMs by grounding responses in retrieved documents.
        Architecture: documents are chunked and embedded into a vector database. At query time, the query is
        embedded and similar documents are retrieved using cosine similarity or approximate nearest neighbor search.
        Retrieved documents are added to the LLM context as grounding. Key components: document chunking strategy,
        embedding model choice, vector database (Chroma, Pinecone, FAISS, Weaviate), retrieval strategy
        (dense, sparse, hybrid), reranking. Advanced RAG: HyDE (hypothetical document embeddings), multi-query
        retrieval, self-RAG with reflection. Evaluation metrics: faithfulness, answer relevance, context recall.""",
        "prerequisites": ["transformers_nlp", "deep_learning"],
        "resources": [
            {"title": "LangChain RAG Tutorial", "url": "https://python.langchain.com/docs/tutorials/rag/"},
        ],
        "duration_hours": 30,
    },
    {
        "id": "prompt_engineering",
        "topic": "AI Engineering",
        "level": "intermediate",
        "content": """Prompt engineering is the practice of crafting effective inputs to language models to elicit
        desired outputs. Techniques: zero-shot prompting (direct instruction), few-shot prompting (examples in
        the prompt), chain-of-thought prompting (ask the model to reason step by step), role prompting (assign
        a persona), and structured output prompting (request JSON or specific formats). System prompts set
        overall behavior and constraints. Temperature controls randomness (0=deterministic, 1=creative).
        Top-p (nucleus sampling) and top-k control token selection diversity. Context window management:
        chunking long documents, summarization, and retrieval to fit within limits. Prompt injection attacks
        and defenses. Evaluation: LLM-as-judge, human evaluation, automated metrics.""",
        "prerequisites": ["python_intro"],
        "resources": [
            {"title": "Anthropic Prompt Engineering Guide", "url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview"},
        ],
        "duration_hours": 10,
    },
]

QUIZ_BANK = {
    "python_intro": [
        {"q": "What does Python use to define code blocks?", "options": ["Curly braces", "Indentation", "Parentheses", "Brackets"], "answer": 1},
        {"q": "Which of these is a valid Python variable declaration?", "options": ["int x = 5", "var x = 5", "x = 5", "declare x = 5"], "answer": 2},
        {"q": "Python is best described as:", "options": ["Compiled, statically typed", "Interpreted, dynamically typed", "Compiled, dynamically typed", "Interpreted, statically typed"], "answer": 1},
    ],
    "linear_algebra": [
        {"q": "What does the dot product of two vectors measure?", "options": ["Their sum", "Their similarity/projection", "Their cross product", "Their magnitude difference"], "answer": 1},
        {"q": "What is a matrix transpose?", "options": ["Flipping rows and columns", "Multiplying by -1", "Inverting the matrix", "Rotating 90 degrees"], "answer": 0},
    ],
    "ml_foundations": [
        {"q": "What is overfitting?", "options": ["Model performs well on training and test data", "Model memorizes training data but fails on new data", "Model is too simple to learn", "Model has too few parameters"], "answer": 1},
        {"q": "Which algorithm is supervised learning?", "options": ["K-Means Clustering", "PCA", "Linear Regression", "DBSCAN"], "answer": 2},
        {"q": "What is the purpose of a validation set?", "options": ["To train the model", "To tune hyperparameters and prevent overfitting", "To evaluate final model performance", "To generate more training data"], "answer": 1},
    ],
    "deep_learning": [
        {"q": "What activation function is most commonly used in hidden layers?", "options": ["Sigmoid", "Tanh", "ReLU", "Softmax"], "answer": 2},
        {"q": "What is backpropagation?", "options": ["Forward pass through network", "Algorithm using chain rule to compute gradients", "Weight initialization method", "Regularization technique"], "answer": 1},
    ],
    "statistics_basics": [
        {"q": "What does p-value represent?", "options": ["Probability the hypothesis is true", "Probability of observed result given null hypothesis", "Confidence level", "Effect size"], "answer": 1},
        {"q": "What is the Central Limit Theorem?", "options": ["All data is normally distributed", "Sample means approach normal distribution as n increases", "Variance always equals mean", "Standard deviation equals variance"], "answer": 1},
    ],
}


def seed_knowledge_base(store: VectorStore) -> int:
    count = 0
    for entry in KNOWLEDGE_BASE:
        chunks = chunk_text(entry["content"], chunk_size=200, overlap=30)
        for i, chunk in enumerate(chunks):
            doc_id = f"{entry['id']}_chunk_{i}"
            metadata = {
                "source_id": entry["id"],
                "topic": entry["topic"],
                "level": entry["level"],
                "prerequisites": entry["prerequisites"],
                "resources": entry.get("resources", []),
                "duration_hours": entry.get("duration_hours", 0),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "title": entry["id"].replace("_", " ").title(),
            }
            store.add_document(doc_id, chunk, metadata)
            count += 1
    logger.info(f"Seeded {count} document chunks from {len(KNOWLEDGE_BASE)} knowledge entries")
    return count


def get_full_entry(entry_id: str) -> Optional[Dict]:
    for entry in KNOWLEDGE_BASE:
        if entry["id"] == entry_id:
            return entry
    return None


def get_quiz_questions(topic_id: str) -> List[Dict]:
    return QUIZ_BANK.get(topic_id, [])


def get_all_topics() -> List[Dict]:
    return [
        {
            "id": e["id"],
            "topic": e["topic"],
            "level": e["level"],
            "prerequisites": e["prerequisites"],
            "duration_hours": e.get("duration_hours", 0),
        }
        for e in KNOWLEDGE_BASE
    ]

