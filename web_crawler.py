#!/usr/bin/env python3

import requests
import threading
import time
import json
import sqlite3
import re
import urllib.parse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from collections import deque, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
import logging

@dataclass
class CrawlStats:
    urls_crawled: int = 0
    urls_discovered: int = 0
    keywords_extracted: int = 0
    start_time: float = 0
    errors: int = 0
    
class WebArchive:
    
    def __init__(self, db_path: str = "web_archive.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                keywords TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status_code INTEGER,
                content_type TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                urls_crawled INTEGER,
                urls_discovered INTEGER,
                keywords_extracted INTEGER,
                crawl_speed REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_page(self, url: str, title: str, content: str, keywords: List[str], 
                   status_code: int, content_type: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        keywords_str = json.dumps(keywords)
        
        cursor.execute('''
            INSERT OR REPLACE INTO pages 
            (url, title, content, keywords, status_code, content_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (url, title, content, keywords_str, status_code, content_type))
        
        conn.commit()
        conn.close()
    
    def store_stats(self, stats: CrawlStats):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        crawl_time = time.time() - stats.start_time
        crawl_speed = stats.urls_crawled / (crawl_time / 60) if crawl_time > 0 else 0
        
        cursor.execute('''
            INSERT INTO crawl_stats 
            (urls_crawled, urls_discovered, keywords_extracted, crawl_speed)
            VALUES (?, ?, ?, ?)
        ''', (stats.urls_crawled, stats.urls_discovered, stats.keywords_extracted, crawl_speed))
        
        conn.commit()
        conn.close()

class KeywordExtractor:
    
    def __init__(self):
        # Common stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
            'our', 'their', 'from', 'as', 'not', 'can', 'than', 'so', 'if', 'when'
        }
    
    def extract_keywords(self, text: str, max_keywords: int = 50) -> List[str]:
        if not text:
            return []
        
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        keywords = [word for word in words 
                   if len(word) > 2 and word not in self.stop_words]
        
        keyword_counts = Counter(keywords)
        return [word for word, _ in keyword_counts.most_common(max_keywords)]

class WebCrawler:
    
    def __init__(self, max_urls: int = 1000, max_threads: int = 5, delay: float = 1.0):
        self.max_urls = max_urls
        self.max_threads = max_threads
        self.delay = delay
        
        self.url_queue = deque()
        self.visited_urls: Set[str] = set()
        self.discovered_urls: Set[str] = set()
        
        self.archive = WebArchive()
        self.keyword_extractor = KeywordExtractor()
        self.stats = CrawlStats()
        
        self.lock = threading.Lock()
        self.running = True
        
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Educational-WebCrawler/1.0 (Educational Purpose)'
        })
    
    def is_valid_url(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
        except:
            return False
    
    def normalize_url(self, url: str, base_url: str = None) -> str:
        if base_url:
            url = urllib.parse.urljoin(base_url, url)
        
        parsed = urllib.parse.urlparse(url)
        # Remove fragment and normalize
        normalized = urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, 
             parsed.params, parsed.query, '')
        )
        return normalized.rstrip('/')
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            normalized_url = self.normalize_url(href, base_url)
            
            if (self.is_valid_url(normalized_url) and 
                normalized_url not in self.visited_urls and
                normalized_url not in self.discovered_urls):
                links.append(normalized_url)
                self.discovered_urls.add(normalized_url)
        
        return links
    
    def crawl_page(self, url: str) -> Optional[Dict]:
        try:
            self.logger.info(f"Crawling: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('content-type', ''):
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ''
            
            for script in soup(["script", "style"]):
                script.decompose()
            text_content = soup.get_text()
            
            keywords = self.keyword_extractor.extract_keywords(text_content)
            
            links = self.extract_links(soup, url)
            
            return {
                'url': url,
                'title': title,
                'content': text_content[:5000],  # Limit content size
                'keywords': keywords,
                'links': links,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', '')
            }
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            with self.lock:
                self.stats.errors += 1
            return None
    
    def worker_thread(self):
        while self.running:
            with self.lock:
                if not self.url_queue or self.stats.urls_crawled >= self.max_urls:
                    break
                url = self.url_queue.popleft()
                self.visited_urls.add(url)
            
            page_data = self.crawl_page(url)
            
            if page_data:
                self.archive.store_page(
                    page_data['url'], page_data['title'], page_data['content'],
                    page_data['keywords'], page_data['status_code'], 
                    page_data['content_type']
                )
                
                with self.lock:
                    for link in page_data['links']:
                        if (len(self.url_queue) < 10000 and  # Prevent queue overflow
                            link not in self.visited_urls):
                            self.url_queue.append(link)
                    
                    self.stats.urls_crawled += 1
                    self.stats.urls_discovered += len(page_data['links'])
                    self.stats.keywords_extracted += len(page_data['keywords'])
            
            time.sleep(self.delay)  # Respectful crawling delay
    
    def start_crawling(self, seed_urls: List[str]):
        self.logger.info(f"Starting crawl with {len(seed_urls)} seed URLs")
        self.stats.start_time = time.time()
        
        for url in seed_urls:
            normalized = self.normalize_url(url)
            if self.is_valid_url(normalized):
                self.url_queue.append(normalized)
                self.discovered_urls.add(normalized)
        
        threads = []
        for i in range(self.max_threads):
            thread = threading.Thread(target=self.worker_thread)
            thread.start()
            threads.append(thread)
        
        self.monitor_progress()
        
        self.running = False
        for thread in threads:
            thread.join()
        
        self.archive.store_stats(self.stats)
        
        self.logger.info(f"Crawling completed. URLs crawled: {self.stats.urls_crawled}")
    
    def monitor_progress(self):
        start_time = time.time()
        
        while self.running and self.stats.urls_crawled < self.max_urls:
            time.sleep(30)  # Update every 30 seconds
            
            elapsed_time = time.time() - start_time
            crawl_speed = self.stats.urls_crawled / (elapsed_time / 60) if elapsed_time > 0 else 0
            
            self.logger.info(
                f"Progress: {self.stats.urls_crawled}/{self.max_urls} URLs crawled, "
                f"{len(self.url_queue)} in queue, "
                f"{self.stats.keywords_extracted} keywords extracted, "
                f"Speed: {crawl_speed:.2f} pages/minute"
            )
            
            if len(self.url_queue) == 0:
                self.logger.warning("URL queue is empty, stopping crawl")
                break
    
    def generate_statistics_report(self):
        conn = sqlite3.connect(self.archive.db_path)
        
        stats_df = pd.read_sql_query("SELECT * FROM crawl_stats ORDER BY timestamp", conn)
        pages_df = pd.read_sql_query("SELECT * FROM pages", conn)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        if not stats_df.empty:
            ax1.plot(range(len(stats_df)), stats_df['urls_crawled'], 'b-', linewidth=2)
            ax1.set_title('URLs Crawled Over Time')
            ax1.set_xlabel('Time Intervals')
            ax1.set_ylabel('Cumulative URLs Crawled')
            ax1.grid(True)
        
        if not stats_df.empty:
            ax2.plot(range(len(stats_df)), stats_df['crawl_speed'], 'r-', linewidth=2)
            ax2.set_title('Crawl Speed (Pages/Minute)')
            ax2.set_xlabel('Time Intervals')
            ax2.set_ylabel('Pages per Minute')
            ax2.grid(True)
        
        if not stats_df.empty:
            ax3.plot(range(len(stats_df)), stats_df['keywords_extracted'], 'g-', linewidth=2)
            ax3.set_title('Keywords Extracted Over Time')
            ax3.set_xlabel('Time Intervals')
            ax3.set_ylabel('Cumulative Keywords')
            ax3.grid(True)
        
        if not pages_df.empty and not stats_df.empty:
            ratios = []
            for _, row in stats_df.iterrows():
                if row['urls_discovered'] > 0:
                    ratio = row['urls_crawled'] / row['urls_discovered']
                    ratios.append(ratio)
                else:
                    ratios.append(0)
            
            ax4.plot(range(len(ratios)), ratios, 'm-', linewidth=2)
            ax4.set_title('Crawl Ratio (URLs Crawled / URLs Discovered)')
            ax4.set_xlabel('Time Intervals')
            ax4.set_ylabel('Ratio')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('crawl_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        conn.close()
        
        return stats_df, pages_df

def main():
    parser = argparse.ArgumentParser(description='Advanced Web Crawler')
    parser.add_argument('--seed-urls', nargs='+', 
                       default=['https://cc.gatech.edu'],
                       help='Seed URLs to start crawling')
    parser.add_argument('--max-urls', type=int, default=1000,
                       help='Maximum number of URLs to crawl')
    parser.add_argument('--threads', type=int, default=5,
                       help='Number of crawler threads')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    crawler = WebCrawler(
        max_urls=args.max_urls,
        max_threads=args.threads,
        delay=args.delay
    )
    
    try:
        crawler.start_crawling(args.seed_urls)
        
        print("\nGenerating statistics report...")
        stats_df, pages_df = crawler.generate_statistics_report()
        
        print(f"\n=== CRAWL SUMMARY ===")
        print(f"Total URLs crawled: {crawler.stats.urls_crawled}")
        print(f"Total URLs discovered: {crawler.stats.urls_discovered}")
        print(f"Total keywords extracted: {crawler.stats.keywords_extracted}")
        print(f"Total errors: {crawler.stats.errors}")
        
        elapsed_time = time.time() - crawler.stats.start_time
        print(f"Total crawl time: {elapsed_time:.2f} seconds")
        print(f"Average crawl speed: {crawler.stats.urls_crawled / (elapsed_time / 60):.2f} pages/minute")
        
        current_speed = crawler.stats.urls_crawled / (elapsed_time / 60)  # pages/minute
        if current_speed > 0:
            time_10m = (10_000_000 / current_speed) / (60 * 24)  # days
            time_1b = (1_000_000_000 / current_speed) / (60 * 24)  # days
            
            print(f"\n=== PREDICTIONS ===")
            print(f"Time to crawl 10 million pages: {time_10m:.1f} days")
            print(f"Time to crawl 1 billion pages: {time_1b:.1f} days")
        
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user")
        crawler.running = False

if __name__ == "__main__":
    main()
