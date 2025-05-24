from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import re
import logging
from logging.handlers import RotatingFileHandler
import json
import csv
from io import StringIO
from urllib.parse import urlparse, urljoin
import time
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import validators
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, EmailStr
import uvicorn
from playwright.sync_api import sync_playwright
import aiohttp
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import yaml
import xml.etree.ElementTree as ET
import markdown
from io import BytesIO
from supabase_config import Auth

# Load environment variables
load_dotenv()

# Initialize FastAPI app
fastapi_app = FastAPI(title="Parcify API", description="AI-Powered Web Data Extraction API")

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class ParseRequest(BaseModel):
    url: HttpUrl
    type: str = "h1"
    query: Optional[str] = None
    options: Optional[Dict[str, bool]] = None

class ParseResponse(BaseModel):
    status: str
    data: Any
    metadata: Dict[str, Any]
    insights: Optional[str] = None

# Initialize AI models
sentiment_analyzer = pipeline("sentiment-analysis")
text_classifier = pipeline("text-classification")
ner_pipeline = pipeline("ner")

# Initialize LangChain components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
embeddings = HuggingFaceEmbeddings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('parcify.log', maxBytes=10000, backupCount=1),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Создаем экземпляр Flask-приложения с названием Parcify
app = Flask("Parcify")

# Hugging Face API configuration
HUGGING_FACE_API_URL = os.getenv('HUGGING_FACE_API_URL')
HUGGING_FACE_API_TOKEN = os.getenv('HUGGING_FACE_API_TOKEN')

# Admin email list
ADMIN_EMAILS = [
    'your@email.com',  # Replace with your real admin email
]

def is_admin(user):
    return user and user.get('email') in ADMIN_EMAILS

class DataValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        return validators.email(email)
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        return bool(re.match(r'^\+?[\d\s-()]{10,}$', phone))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        return validators.url(url)
    
    @staticmethod
    def validate_price(price: float) -> bool:
        return isinstance(price, (int, float)) and price >= 0
    
    @staticmethod
    def validate_rating(rating: float) -> bool:
        return isinstance(rating, (int, float)) and 0 <= rating <= 5

class DataAnalyzer:
    @staticmethod
    def analyze_prices(prices: List[float]) -> Dict[str, Any]:
        if not prices:
            return {}
        
        return {
            'min': min(prices),
            'max': max(prices),
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices)
        }
    
    @staticmethod
    def analyze_ratings(ratings: List[float]) -> Dict[str, Any]:
        if not ratings:
            return {}
        
        return {
            'average': np.mean(ratings),
            'distribution': {
                '5_star': len([r for r in ratings if r >= 4.5]),
                '4_star': len([r for r in ratings if 3.5 <= r < 4.5]),
                '3_star': len([r for r in ratings if 2.5 <= r < 3.5]),
                '2_star': len([r for r in ratings if 1.5 <= r < 2.5]),
                '1_star': len([r for r in ratings if r < 1.5])
            }
        }
    
    @staticmethod
    def analyze_products(products: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not products:
            return {}
        
        prices = [p.get('price', 0) for p in products if 'price' in p]
        return {
            'total_products': len(products),
            'price_analysis': DataAnalyzer.analyze_prices(prices),
            'categories': list(set(p.get('category', '') for p in products if 'category' in p))
        }

class MetadataExtractor:
    @staticmethod
    def extract_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        # Extract title
        title = soup.find('title')
        if title:
            metadata['title'] = title.text.strip()
        
        # Extract Open Graph tags
        og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
        metadata['og'] = {tag['property'][3:]: tag['content'] for tag in og_tags if 'content' in tag.attrs}
        
        # Extract Twitter Card tags
        twitter_tags = soup.find_all('meta', name=re.compile(r'^twitter:'))
        metadata['twitter'] = {tag['name'][8:]: tag['content'] for tag in twitter_tags if 'content' in tag.attrs}
        
        return metadata

class AIInsightsGenerator:
    @staticmethod
    def generate_insights(data: Dict[str, Any], data_type: str) -> str:
        insights = []
        
        if data_type == 'prices':
            if 'price_analysis' in data:
                analysis = data['price_analysis']
                insights.append(f"Price Range: ${analysis['min']:.2f} - ${analysis['max']:.2f}")
                insights.append(f"Average Price: ${analysis['mean']:.2f}")
                insights.append(f"Price Standard Deviation: ${analysis['std']:.2f}")
        
        elif data_type == 'reviews':
            if 'rating_analysis' in data:
                analysis = data['rating_analysis']
                insights.append(f"Average Rating: {analysis['average']:.1f}/5.0")
                insights.append("Rating Distribution:")
                for rating, count in analysis['distribution'].items():
                    insights.append(f"- {rating}: {count} reviews")
        
        elif data_type == 'products':
            if 'product_analysis' in data:
                analysis = data['product_analysis']
                insights.append(f"Total Products: {analysis['total_products']}")
                if 'categories' in analysis:
                    insights.append(f"Categories: {', '.join(analysis['categories'])}")
        
        return "\n".join(insights)

def analyze_query(query: str) -> str:
    """Analyze user query using Hugging Face API to determine the type of data to extract."""
    try:
        headers = {
            "Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Given the following query about web scraping, determine what type of data should be extracted.
        Query: {query}
        Choose from: headings, links, images, prices, tables, products, reviews, contact_info
        Answer with just one word from the options."""
        
        response = requests.post(
            HUGGING_FACE_API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        
        if response.status_code == 200:
            result = response.json()
            predicted_type = result[0]['generated_text'].strip().lower()
            
            type_mapping = {
                'headings': 'h1',
                'links': 'links',
                'images': 'images',
                'prices': 'prices',
                'tables': 'tables',
                'products': 'products',
                'reviews': 'reviews',
                'contact_info': 'contact_info'
            }
            
            return type_mapping.get(predicted_type, 'h1')
        else:
            logger.error(f"Hugging Face API error: {response.status_code}")
            return 'h1'
            
    except Exception as e:
        logger.error(f"Error in query analysis: {str(e)}")
        return 'h1'

def extract_products(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract product information using AI-powered pattern recognition."""
    products = []
    
    # Look for common product patterns
    product_containers = soup.find_all(['div', 'article'], class_=re.compile(r'(product|item|card)', re.IGNORECASE))
    
    for container in product_containers:
        product = {}
        
        # Extract product name
        name_elem = container.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'div'], 
                                 class_=re.compile(r'(title|name|product-name)', re.IGNORECASE))
        if name_elem:
            product['name'] = name_elem.text.strip()
        
        # Extract price
        price_elem = container.find(['span', 'div'], class_=re.compile(r'(price|cost|amount)', re.IGNORECASE))
        if price_elem:
            price_match = re.search(r'\d+[\.,]?\d*', price_elem.text)
            if price_match:
                product['price'] = float(price_match.group().replace(',', '.'))
        
        # Extract image
        img_elem = container.find('img')
        if img_elem and img_elem.get('src'):
            product['image'] = img_elem['src']
            product['image_alt'] = img_elem.get('alt', '')
        
        # Extract description
        desc_elem = container.find(['p', 'div'], class_=re.compile(r'(description|details|info)', re.IGNORECASE))
        if desc_elem:
            product['description'] = desc_elem.text.strip()
        
        # Extract category
        category_elem = container.find(['span', 'div'], class_=re.compile(r'(category|type)', re.IGNORECASE))
        if category_elem:
            product['category'] = category_elem.text.strip()
        
        if product:
            products.append(product)
    
    return products

def extract_reviews(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract review information using AI-powered pattern recognition."""
    reviews = []
    
    # Look for common review patterns
    review_containers = soup.find_all(['div', 'article'], class_=re.compile(r'(review|comment|rating)', re.IGNORECASE))
    
    for container in review_containers:
        review = {}
        
        # Extract rating
        rating_elem = container.find(['span', 'div'], class_=re.compile(r'(rating|stars|score)', re.IGNORECASE))
        if rating_elem:
            rating_match = re.search(r'\d+[\.,]?\d*', rating_elem.text)
            if rating_match:
                review['rating'] = float(rating_match.group().replace(',', '.'))
        
        # Extract review text
        text_elem = container.find(['p', 'div'], class_=re.compile(r'(text|content|body)', re.IGNORECASE))
        if text_elem:
            review['text'] = text_elem.text.strip()
        
        # Extract author
        author_elem = container.find(['span', 'div'], class_=re.compile(r'(author|user|name)', re.IGNORECASE))
        if author_elem:
            review['author'] = author_elem.text.strip()
        
        # Extract date
        date_elem = container.find(['span', 'div'], class_=re.compile(r'(date|time)', re.IGNORECASE))
        if date_elem:
            review['date'] = date_elem.text.strip()
        
        if review:
            reviews.append(review)
    
    return reviews

def extract_contact_info(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract contact information using AI-powered pattern recognition."""
    contact_info = {}
    
    # Look for email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, str(soup))
    if emails:
        contact_info['emails'] = list(set(emails))
    
    # Look for phone numbers
    phone_pattern = r'\+?[\d\s-()]{10,}'
    phones = re.findall(phone_pattern, str(soup))
    if phones:
        contact_info['phones'] = list(set(phones))
    
    # Look for social media links
    social_media = {
        'facebook': r'facebook\.com/[^"\']+',
        'twitter': r'twitter\.com/[^"\']+',
        'linkedin': r'linkedin\.com/[^"\']+',
        'instagram': r'instagram\.com/[^"\']+'
    }
    
    for platform, pattern in social_media.items():
        matches = re.findall(pattern, str(soup))
        if matches:
            contact_info[platform] = list(set(matches))
    
    return contact_info

def follow_internal_links(soup: BeautifulSoup, base_url: str, max_depth: int = 2) -> List[Dict[str, Any]]:
    """Follow internal links and extract data from linked pages."""
    results = []
    visited_urls = set()
    
    def extract_from_url(url: str, depth: int = 0) -> None:
        if depth > max_depth or url in visited_urls:
            return
        
        visited_urls.add(url)
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if response.status_code == 200:
                page_soup = BeautifulSoup(response.content, 'html.parser')
                results.append({
                    'url': url,
                    'title': page_soup.find('title').text.strip() if page_soup.find('title') else '',
                    'content': page_soup.get_text()[:500] + '...'  # First 500 characters
                })
                
                # Find and follow internal links
                if depth < max_depth:
                    for link in page_soup.find_all('a', href=True):
                        href = link['href']
                        absolute_url = urljoin(base_url, href)
                        if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                            extract_from_url(absolute_url, depth + 1)
        except Exception as e:
            logger.error(f"Error following link {url}: {str(e)}")
    
    extract_from_url(base_url)
    return results

def validate_data(data: Any, data_type: str) -> Dict[str, Any]:
    """Validate extracted data based on its type."""
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if data_type == 'contact_info':
        if 'emails' in data:
            for email in data['emails']:
                if not DataValidator.validate_email(email):
                    validation_results['issues'].append(f"Invalid email: {email}")
        if 'phones' in data:
            for phone in data['phones']:
                if not DataValidator.validate_phone(phone):
                    validation_results['issues'].append(f"Invalid phone: {phone}")
    
    elif data_type == 'products':
        for product in data:
            if 'price' in product and not DataValidator.validate_price(product['price']):
                validation_results['issues'].append(f"Invalid price in product: {product.get('name', 'Unknown')}")
    
    elif data_type == 'reviews':
        for review in data:
            if 'rating' in review and not DataValidator.validate_rating(review['rating']):
                validation_results['issues'].append(f"Invalid rating in review by: {review.get('author', 'Unknown')}")
    
    validation_results['is_valid'] = len(validation_results['issues']) == 0
    return validation_results

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_tables(soup):
    tables = []
    for table in soup.find_all('table'):
        table_data = []
        for row in table.find_all('tr'):
            row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if row_data:
                table_data.append(row_data)
        if table_data:
            tables.append(table_data)
    return tables

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse():
    start_time = time.time()
    data = request.json
    url = data.get('url')
    parse_type = data.get('type', 'h1')
    query = data.get('query', '')
    options = data.get('options', {})

    if not url:
        logger.error("No URL provided")
        return jsonify({"error": "URL is required"}), 400

    if not is_valid_url(url):
        logger.error(f"Invalid URL provided: {url}")
        return jsonify({"error": "Invalid URL format"}), 400

    # If query is provided, analyze it to determine parse type
    if query:
        parse_type = analyze_query(query)
        logger.info(f"Query analysis result: {parse_type}")

    try:
        logger.info(f"Processing request for URL: {url}, type: {parse_type}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        result = []
        metadata = {}
        insights = None

        # Extract data based on type
        if parse_type == 'h1':
            result = [h.text.strip() for h in soup.find_all('h1')]
        elif parse_type == 'links':
            result = [{'text': a.text.strip(), 'href': a['href']} 
                     for a in soup.find_all('a', href=True)]
        elif parse_type == 'images':
            result = [{'src': img['src'], 'alt': img.get('alt', '')} 
                     for img in soup.find_all('img', src=True)]
        elif parse_type == 'prices':
            price_tags = soup.find_all(['span', 'div'], 
                                     class_=re.compile(r'(price|cost|amount)', re.IGNORECASE))
            prices = []
            for tag in price_tags:
                matches = re.findall(r'\d+[\.,]?\d*', tag.text)
                for price in matches:
                    try:
                        prices.append(float(price.replace(',', '.')))
                    except ValueError:
                        continue
            result = prices
        elif parse_type == 'tables':
            result = extract_tables(soup)
        elif parse_type == 'products':
            result = extract_products(soup)
        elif parse_type == 'reviews':
            result = extract_reviews(soup)
        elif parse_type == 'contact_info':
            result = extract_contact_info(soup)
        else:
            logger.error(f"Invalid parse type: {parse_type}")
            return jsonify({"error": "Invalid parse type"}), 400

        # Extract metadata if requested
        if options.get('extractMetadata'):
            metadata = MetadataExtractor.extract_metadata(soup)

        # Follow internal links if requested
        if options.get('followLinks'):
            linked_data = follow_internal_links(soup, url)
            result.extend(linked_data)

        # Validate data if requested
        validation_results = None
        if options.get('validateData'):
            validation_results = validate_data(result, parse_type)

        # Generate insights if requested
        if options.get('generateInsights'):
            analysis = {}
            if parse_type == 'prices':
                analysis = DataAnalyzer.analyze_prices(result)
            elif parse_type == 'reviews':
                analysis = DataAnalyzer.analyze_ratings([r.get('rating', 0) for r in result])
            elif parse_type == 'products':
                analysis = DataAnalyzer.analyze_products(result)
            
            insights = AIInsightsGenerator.generate_insights(analysis, parse_type)

        processing_time = time.time() - start_time
        logger.info(f"Successfully processed request in {processing_time:.2f} seconds")
        
        response_data = {
            "status": "success",
            "data": result,
            "metadata": {
                "processing_time": f"{processing_time:.2f} seconds",
                "items_found": len(result),
                "parse_type": parse_type,
                "ai_enhanced": True,
                "validation": validation_results,
                "page_metadata": metadata
            }
        }

        if insights:
            response_data["insights"] = insights

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for URL {url}: {str(e)}")
        return jsonify({"error": f"Failed to fetch URL: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Add new data processing functions
class DataProcessor:
    @staticmethod
    def process_data(data: Any, format: str) -> str:
        if format == 'csv':
            return DataProcessor.to_csv(data)
        elif format == 'excel':
            return DataProcessor.to_excel(data)
        elif format == 'xml':
            return DataProcessor.to_xml(data)
        elif format == 'yaml':
            return DataProcessor.to_yaml(data)
        elif format == 'markdown':
            return DataProcessor.to_markdown(data)
        elif format == 'html':
            return DataProcessor.to_html(data)
        else:
            return json.dumps(data, indent=2)

    @staticmethod
    def to_csv(data: List[Dict]) -> str:
        if not data:
            return ""
        df = pd.DataFrame(data)
        return df.to_csv(index=False)

    @staticmethod
    def to_excel(data: List[Dict]) -> bytes:
        if not data:
            return b""
        df = pd.DataFrame(data)
        output = BytesIO()
        df.to_excel(output, index=False)
        return output.getvalue()

    @staticmethod
    def to_xml(data: List[Dict]) -> str:
        root = ET.Element("data")
        for item in data:
            record = ET.SubElement(root, "record")
            for key, value in item.items():
                field = ET.SubElement(record, key)
                field.text = str(value)
        return ET.tostring(root, encoding='unicode')

    @staticmethod
    def to_yaml(data: List[Dict]) -> str:
        return yaml.dump(data, default_flow_style=False)

    @staticmethod
    def to_markdown(data: List[Dict]) -> str:
        if not data:
            return ""
        headers = list(data[0].keys())
        md = ["| " + " | ".join(headers) + " |",
              "| " + " | ".join(["---"] * len(headers)) + " |"]
        for item in data:
            md.append("| " + " | ".join(str(item[h]) for h in headers) + " |")
        return "\n".join(md)

    @staticmethod
    def to_html(data: List[Dict]) -> str:
        if not data:
            return ""
        headers = list(data[0].keys())
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <title>Parcify Export</title>",
            "  <style>",
            "    table { border-collapse: collapse; width: 100%; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background-color: #f2f2f2; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <table>",
            "    <thead>",
            "      <tr>",
            "        " + "".join(f"<th>{h}</th>" for h in headers),
            "      </tr>",
            "    </thead>",
            "    <tbody>"
        ]
        for item in data:
            html.append(
                "      <tr>" +
                "".join(f"<td>{item[h]}</td>" for h in headers) +
                "      </tr>"
            )
        html.extend([
            "    </tbody>",
            "  </table>",
            "</body>",
            "</html>"
        ])
        return "\n".join(html)

# Add semantic search functionality
class SemanticSearch:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.metadata_store = {}
        self.search_history = []

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        texts = self.text_splitter.split_documents(documents)
        if metadata:
            for i, doc in enumerate(texts):
                doc.metadata.update(metadata[i] if i < len(metadata) else {})
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        self.metadata_store = {doc.metadata.get('id', i): doc.metadata for i, doc in enumerate(texts)}

    def search(self, query: str, k: int = 5, filters: Optional[Dict] = None, 
              sort_by: Optional[str] = None, sort_order: str = 'desc') -> List[Dict[str, Any]]:
        if not self.vector_store:
            return []
        
        # Store search history
        self.search_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'filters': filters,
            'results_count': k
        })
        
        # Perform semantic search
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Process results
        processed_results = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            }
            
            # Apply filters if provided
            if filters:
                if all(result['metadata'].get(k) == v for k, v in filters.items()):
                    processed_results.append(result)
            else:
                processed_results.append(result)
        
        # Sort results if requested
        if sort_by:
            processed_results.sort(
                key=lambda x: x['metadata'].get(sort_by, 0),
                reverse=(sort_order == 'desc')
            )
        
        return processed_results

    def get_search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.search_history[-limit:]

    def get_metadata_stats(self) -> Dict[str, Any]:
        if not self.metadata_store:
            return {}
        
        stats = {}
        for metadata in self.metadata_store.values():
            for key, value in metadata.items():
                if key not in stats:
                    stats[key] = set()
                stats[key].add(value)
        
        return {k: list(v) for k, v in stats.items()}

    def get_similar_documents(self, doc_id: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vector_store or doc_id not in self.metadata_store:
            return []
        
        doc_metadata = self.metadata_store[doc_id]
        doc_content = self.vector_store.docstore.search(doc_id)
        
        if not doc_content:
            return []
        
        similar_docs = self.vector_store.similarity_search_with_score(
            doc_content.page_content,
            k=k + 1  # +1 because the document itself will be included
        )
        
        return [
            {
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            }
            for doc, score in similar_docs
            if doc.metadata.get('id') != doc_id  # Exclude the original document
        ]

    def get_document_clusters(self, n_clusters: int = 5) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return []
        
        # Get all document embeddings
        embeddings = self.vector_store.index.reconstruct_n(0, self.vector_store.index.ntotal)
        
        # Perform clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group documents by cluster
        cluster_results = []
        for i in range(n_clusters):
            cluster_docs = []
            for j, cluster_id in enumerate(clusters):
                if cluster_id == i:
                    doc = self.vector_store.docstore.search(j)
                    if doc:
                        cluster_docs.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
            
            if cluster_docs:
                cluster_results.append({
                    "cluster_id": i,
                    "size": len(cluster_docs),
                    "documents": cluster_docs
                })
        
        return cluster_results

# Initialize semantic search
semantic_search = SemanticSearch()

# Add new FastAPI endpoints
@fastapi_app.post("/api/semantic-search")
async def semantic_search_endpoint(
    query: str,
    k: int = 5,
    filters: Optional[Dict] = None,
    sort_by: Optional[str] = None,
    sort_order: str = 'desc'
):
    try:
        results = semantic_search.search(
            query=query,
            k=k,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return {
            "results": results,
            "metadata": {
                "total_results": len(results),
                "query": query,
                "filters_applied": filters,
                "sorting": {"by": sort_by, "order": sort_order} if sort_by else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add new endpoints for enhanced semantic search features
@fastapi_app.get("/api/search-history")
async def get_search_history(limit: int = 10):
    try:
        history = semantic_search.get_search_history(limit)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/metadata-stats")
async def get_metadata_stats():
    try:
        stats = semantic_search.get_metadata_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/similar-documents/{doc_id}")
async def get_similar_documents(doc_id: str, k: int = 5):
    try:
        similar_docs = semantic_search.get_similar_documents(doc_id, k)
        return {"similar_documents": similar_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/document-clusters")
async def get_document_clusters(n_clusters: int = 5):
    try:
        clusters = semantic_search.get_document_clusters(n_clusters)
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/api/process-data")
async def process_data_endpoint(data: List[Dict], format: str):
    try:
        processed_data = DataProcessor.process_data(data, format)
        return {"data": processed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update the parse endpoint to include semantic search
@fastapi_app.post("/api/parse", response_model=ParseResponse)
async def parse_endpoint(request: ParseRequest):
    try:
        # Use existing parse function but with async support
        result = await parse_async(request.url, request.type, request.query, request.options)
        
        # Add documents to semantic search
        if isinstance(result["data"], list):
            documents = [str(item) for item in result["data"]]
            semantic_search.add_documents(documents)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add new FastAPI endpoints
@fastapi_app.post("/api/analyze-sentiment")
async def analyze_sentiment(text: str):
    try:
        result = sentiment_analyzer(text)
        return {"sentiment": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/api/extract-entities")
async def extract_entities(text: str):
    try:
        entities = ner_pipeline(text)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/api/classify-text")
async def classify_text(text: str):
    try:
        classification = text_classifier(text)
        return {"classification": classification[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add async parsing function
async def parse_async(url: str, parse_type: str, query: str, options: Dict[str, bool]) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(str(url)) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Use existing extraction logic but with async support
            result = await extract_data_async(soup, parse_type)
            
            # Add AI-powered analysis
            if options.get('generateInsights'):
                result = await enhance_with_ai(result, parse_type)
            
            return {
                "status": "success",
                "data": result,
                "metadata": {
                    "processing_time": "0.0",
                    "items_found": len(result),
                    "parse_type": parse_type,
                    "ai_enhanced": True
                }
            }

async def extract_data_async(soup: BeautifulSoup, parse_type: str) -> Any:
    # Implement async versions of existing extraction functions
    if parse_type == 'products':
        return await extract_products_async(soup)
    elif parse_type == 'reviews':
        return await extract_reviews_async(soup)
    # ... implement other async extraction methods ...
    return []

async def enhance_with_ai(data: Any, data_type: str) -> Any:
    if data_type == 'reviews':
        # Add sentiment analysis to reviews
        for review in data:
            if 'text' in review:
                sentiment = sentiment_analyzer(review['text'])[0]
                review['sentiment'] = sentiment
    elif data_type == 'products':
        # Add text classification to product descriptions
        for product in data:
            if 'description' in product:
                classification = text_classifier(product['description'])[0]
                product['category_ai'] = classification
    return data

# Add new Pydantic models for authentication
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordUpdateRequest(BaseModel):
    token: str
    new_password: str

# Add authentication endpoints
@fastapi_app.post("/api/auth/signup")
async def signup(request: SignUpRequest):
    try:
        response = await Auth.sign_up(request.email, request.password)
        return {"status": "success", "message": "User registered successfully", "data": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.post("/api/auth/signin")
async def signin(request: SignInRequest):
    try:
        response = await Auth.sign_in(request.email, request.password)
        return {"status": "success", "message": "User signed in successfully", "data": response}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@fastapi_app.post("/api/auth/signout")
async def signout(token: str):
    try:
        response = await Auth.sign_out(token)
        return {"status": "success", "message": "User signed out successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.get("/api/auth/user")
async def get_user(token: str):
    try:
        user = await Auth.get_user(token)
        return {"status": "success", "data": user}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@fastapi_app.post("/api/auth/reset-password")
async def reset_password(request: PasswordResetRequest):
    try:
        response = await Auth.reset_password(request.email)
        return {"status": "success", "message": "Password reset email sent"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.post("/api/auth/update-password")
async def update_password(request: PasswordUpdateRequest):
    try:
        response = await Auth.update_password(request.token, request.new_password)
        return {"status": "success", "message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add new Pydantic models for user profile management
class UserProfileRequest(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[dict] = None

# Add new protected endpoints
@fastapi_app.get("/api/user/profile", response_model=UserProfile)
async def get_user_profile(request: Request):
    try:
        user = request.state.user
        profile = await supabase.table('profiles').select('*').eq('id', user.id).single().execute()
        return profile.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.put("/api/user/profile", response_model=UserProfile)
async def update_user_profile(request: Request, profile_data: UserProfileRequest):
    try:
        user = request.state.user
        profile = await supabase.table('profiles').update(profile_data.dict(exclude_unset=True)).eq('id', user.id).execute()
        return profile.data[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.get("/api/user/search-history")
async def get_user_search_history(request: Request, limit: int = 10):
    try:
        user = request.state.user
        history = await supabase.table('search_history').select('*').eq('user_id', user.id).order('timestamp', desc=True).limit(limit).execute()
        return {"history": history.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.post("/api/user/saved-searches")
async def create_saved_search(request: Request, search: SavedSearch):
    try:
        user = request.state.user
        search.user_id = user.id
        result = await supabase.table('saved_searches').insert(search.dict()).execute()
        return {"saved_search": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.get("/api/user/saved-searches")
async def get_saved_searches(request: Request):
    try:
        user = request.state.user
        searches = await supabase.table('saved_searches').select('*').eq('user_id', user.id).execute()
        return {"saved_searches": searches.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@fastapi_app.get("/api/user/export-history")
async def get_export_history(request: Request, limit: int = 10):
    try:
        user = request.state.user
        history = await supabase.table('export_history').select('*').eq('user_id', user.id).order('timestamp', desc=True).limit(limit).execute()
        return {"history": history.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Update existing endpoints to use authentication middleware
@fastapi_app.post("/api/parse", response_model=ParseResponse)
async def parse_endpoint(request: Request, parse_request: ParseRequest):
    try:
        # Verify user is authenticated
        user = request.state.user
        
        # Use existing parse function but with async support
        result = await parse_async(parse_request.url, parse_request.type, parse_request.query, parse_request.options)
        
        # Add documents to semantic search
        if isinstance(result["data"], list):
            documents = [str(item) for item in result["data"]]
            semantic_search.add_documents(documents)
        
        # Log the parse request
        await supabase.table('parse_history').insert({
            'user_id': user.id,
            'url': str(parse_request.url),
            'parse_type': parse_request.type,
            'query': parse_request.query,
            'timestamp': datetime.now().isoformat()
        }).execute()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add new FastAPI endpoints
@fastapi_app.get('/api/admin/users')
async def admin_get_users(request: Request):
    user = request.state.user
    if not is_admin(user):
        raise HTTPException(status_code=403, detail='Admin access required')
    res = await supabase.table('profiles').select('*').execute()
    return {'users': res.data}

@fastapi_app.get('/api/admin/search-history')
async def admin_get_search_history(request: Request):
    user = request.state.user
    if not is_admin(user):
        raise HTTPException(status_code=403, detail='Admin access required')
    res = await supabase.table('search_history').select('*').execute()
    return {'search_history': res.data}

@fastapi_app.get('/api/admin/export-history')
async def admin_get_export_history(request: Request):
    user = request.state.user
    if not is_admin(user):
        raise HTTPException(status_code=403, detail='Admin access required')
    res = await supabase.table('export_history').select('*').execute()
    return {'export_history': res.data}

@fastapi_app.get('/api/admin/saved-searches')
async def admin_get_saved_searches(request: Request):
    user = request.state.user
    if not is_admin(user):
        raise HTTPException(status_code=403, detail='Admin access required')
    res = await supabase.table('saved_searches').select('*').execute()
    return {'saved_searches': res.data}

if __name__ == '__main__':
    # Run both Flask and FastAPI
    import threading
    
    def run_flask():
        app.run(debug=True, port=5000)
    
    def run_fastapi():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
    
    flask_thread = threading.Thread(target=run_flask)
    fastapi_thread = threading.Thread(target=run_fastapi)
    
    flask_thread.start()
    fastapi_thread.start()
